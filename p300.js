/**
 * p300.js — P300 epoch extraction, artifact rejection, and classification
 * Runs entirely in the browser. No dependencies.
 *
 * Crown channels: CP3(0) C3(1) F5(2) PO3(3) PO4(4) F6(5) C4(6) CP4(7)
 * P300 channels of interest: PO3(3), PO4(4) — posterior parietal
 * Secondary: CP3(0), CP4(7) — centroparietal for broader P300 window
 */

export const SAMPLE_RATE = 256;   // Hz
export const N_CHANNELS = 8;
export const P300_CHANNELS = [3, 4];       // PO3, PO4
export const SECONDARY_CHANNELS = [0, 7]; // CP3, CP4

export const EPOCH_PRE_MS = 100;   // baseline window before stimulus
export const EPOCH_POST_MS = 600;  // capture window after stimulus
export const P300_WIN_START = 250; // ms post-stimulus
export const P300_WIN_END = 550;   // ms post-stimulus
export const ARTIFACT_THRESH_UV = 100; // reject epochs with |signal| > this

export class EEGBuffer {
  constructor(maxDurationMs = 3000) {
    this.samples = [];  // [{t: ms, ch: Float32Array[8]}]
    this.maxDuration = maxDurationMs;
  }

  push(packet) {
    // packet: { data: [[ch0..ch7], ...], startTime: ms, sampleRate: 256 }
    const { data, startTime, sampleRate } = packet;
    const dt = 1000 / sampleRate;

    for (let i = 0; i < data.length; i++) {
      this.samples.push({
        t: startTime + i * dt,
        ch: new Float32Array(data[i]),
      });
    }

    // Prune old samples
    const cutoff = Date.now() - this.maxDuration;
    let trimIdx = 0;
    while (trimIdx < this.samples.length && this.samples[trimIdx].t < cutoff) trimIdx++;
    if (trimIdx > 0) this.samples.splice(0, trimIdx);
  }

  getSamplesInWindow(startMs, endMs) {
    return this.samples.filter(s => s.t >= startMs && s.t <= endMs);
  }

  get length() { return this.samples.length; }
}

/**
 * Extract and baseline-correct an epoch around a stimulus onset.
 * Returns null if insufficient data or artifact detected.
 */
export function extractEpoch(buffer, stimulusTs) {
  const windowStart = stimulusTs - EPOCH_PRE_MS;
  const windowEnd = stimulusTs + EPOCH_POST_MS;
  const samples = buffer.getSamplesInWindow(windowStart, windowEnd);

  const expectedSamples = Math.round((EPOCH_PRE_MS + EPOCH_POST_MS) / 1000 * SAMPLE_RATE);

  if (samples.length < expectedSamples * 0.75) {
    return null; // not enough data yet
  }

  // Separate baseline and post-stimulus samples
  const baseline = samples.filter(s => s.t < stimulusTs);
  const post = samples.filter(s => s.t >= stimulusTs);

  if (baseline.length < 5) return null;

  // Compute baseline mean per channel
  const baselineMeans = new Float32Array(N_CHANNELS);
  for (const s of baseline) {
    for (let c = 0; c < N_CHANNELS; c++) baselineMeans[c] += s.ch[c];
  }
  for (let c = 0; c < N_CHANNELS; c++) baselineMeans[c] /= baseline.length;

  // Build baseline-corrected epoch matrix: [nSamples][nChannels]
  const allSamples = samples.map(s => ({
    t: s.t - stimulusTs, // time relative to stimulus (negative = pre)
    ch: s.ch.map((v, c) => v - baselineMeans[c]),
  }));

  // Artifact rejection: check all channels across full epoch
  for (const s of allSamples) {
    for (let c = 0; c < N_CHANNELS; c++) {
      if (Math.abs(s.ch[c]) > ARTIFACT_THRESH_UV) {
        return null; // artifact — reject epoch
      }
    }
  }

  return allSamples; // [{t: ms_relative, ch: corrected_values}]
}

/**
 * Score an epoch for P300 presence.
 * Returns { amplitude, score, isP300 } where score is 0-1.
 */
export function scoreEpoch(epoch, threshold = 3.0) {
  if (!epoch) return { amplitude: 0, score: 0, isP300: false };

  // Isolate P300 window samples
  const p300Samples = epoch.filter(
    s => s.t >= P300_WIN_START && s.t <= P300_WIN_END
  );

  if (p300Samples.length === 0) return { amplitude: 0, score: 0, isP300: false };

  // Average parietal channels in P300 window
  let sum = 0;
  for (const s of p300Samples) {
    for (const c of P300_CHANNELS) sum += s.ch[c];
  }
  const meanAmplitude = sum / (p300Samples.length * P300_CHANNELS.length);

  // Peak amplitude in window
  let peakAmplitude = -Infinity;
  for (const s of p300Samples) {
    for (const c of P300_CHANNELS) {
      if (s.ch[c] > peakAmplitude) peakAmplitude = s.ch[c];
    }
  }

  // Score: sigmoid-normalized relative to threshold
  const score = Math.max(0, Math.min(1, peakAmplitude / (threshold * 2)));

  return {
    amplitude: Math.round(peakAmplitude * 100) / 100,
    meanAmplitude: Math.round(meanAmplitude * 100) / 100,
    score: Math.round(score * 1000) / 1000,
    isP300: peakAmplitude > threshold,
  };
}

/**
 * Calibration accumulator.
 * Collects target and non-target epochs, computes grand average P300 component,
 * returns recommended threshold.
 */
export class Calibrator {
  constructor() {
    this.targetEpochs = [];
    this.nonTargetEpochs = [];
  }

  addEpoch(epoch, isTarget) {
    if (!epoch) return;
    if (isTarget) this.targetEpochs.push(epoch);
    else this.nonTargetEpochs.push(epoch);
  }

  get targetCount() { return this.targetEpochs.length; }
  get nonTargetCount() { return this.nonTargetEpochs.length; }
  get isReady() { return this.targetEpochs.length >= 10 && this.nonTargetEpochs.length >= 30; }

  /**
   * Compute grand average difference wave and extract recommended threshold.
   * Returns { threshold, p300PeakMs, p300Amplitude, targetAvg, nonTargetAvg }
   */
  compute() {
    if (!this.isReady) return null;

    const avgEpoch = (epochs) => {
      // Find common time points — use first epoch as template
      const template = epochs[0].map(s => ({
        t: s.t,
        ch: new Float32Array(N_CHANNELS),
      }));

      for (const epoch of epochs) {
        for (let i = 0; i < Math.min(template.length, epoch.length); i++) {
          for (let c = 0; c < N_CHANNELS; c++) {
            template[i].ch[c] += epoch[i].ch[c] / epochs.length;
          }
        }
      }
      return template;
    };

    const targetAvg = avgEpoch(this.targetEpochs);
    const nonTargetAvg = avgEpoch(this.nonTargetEpochs);

    // Difference wave on parietal channels in P300 window
    let peakDiff = -Infinity;
    let peakMs = 0;
    for (let i = 0; i < targetAvg.length; i++) {
      const t = targetAvg[i].t;
      if (t < P300_WIN_START || t > P300_WIN_END) continue;
      let diff = 0;
      for (const c of P300_CHANNELS) {
        diff += targetAvg[i].ch[c] - nonTargetAvg[i].ch[c];
      }
      diff /= P300_CHANNELS.length;
      if (diff > peakDiff) { peakDiff = diff; peakMs = t; }
    }

    // Threshold at 50% of peak difference
    const threshold = Math.max(1.5, peakDiff * 0.5);

    return {
      threshold: Math.round(threshold * 100) / 100,
      p300PeakMs: Math.round(peakMs),
      p300Amplitude: Math.round(peakDiff * 100) / 100,
      targetAvg,
      nonTargetAvg,
      nTargets: this.targetEpochs.length,
      nNonTargets: this.nonTargetEpochs.length,
    };
  }

  reset() {
    this.targetEpochs = [];
    this.nonTargetEpochs = [];
  }
}
