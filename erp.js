/**
 * erp.js — N170 + VPP epoch extraction, scoring, and calibration
 *
 * Replaces p300.js for face/body detection use cases.
 *
 * N170: negative deflection at 120–220ms, strongest at posterior temporal sites.
 *       On the Insight: T7(1) and T8(3) — closer to ideal P7/P8 than Crown had.
 *
 * VPP:  positive deflection at 140–220ms at vertex (Cz/Pz).
 *       On the Insight: Pz(2) — parietal midline, optimal VPP site.
 *
 * Combined score uses both components — N170 negativity + VPP positivity.
 *
 * Emotiv Insight channel map (5ch, 128Hz):
 *   Index  Label   Position
 *     0    AF3     Left frontoparietal  (minimal N170, some FAA)
 *     1    T7      Left temporal        ← PRIMARY N170
 *     2    Pz      Parietal midline     ← PRIMARY VPP
 *     3    T8      Right temporal       ← PRIMARY N170
 *     4    AF4     Right frontoparietal (minimal N170, some FAA)
 */

export const SAMPLE_RATE = 128;  // Emotiv Insight is 128Hz
export const N_CHANNELS  = 5;

// Emotiv Insight channel map: [AF3(0), T7(1), Pz(2), T8(3), AF4(4)]
// T7 and T8 are posterior temporal — excellent N170 sites (close to P7/P8)
// Pz is parietal midline — ideal for VPP
export const N170_CHANNELS  = [1, 3];   // T7, T8   — primary N170 (posterior temporal)
export const N170_SECONDARY = [0, 4];   // AF3, AF4 — frontal spread (minimal N170)
export const VPP_CHANNELS   = [2];      // Pz       — parietal midline, optimal VPP

// Timing windows (ms relative to stimulus onset)
export const EPOCH_PRE_MS   = 100;   // baseline window before stimulus
export const EPOCH_POST_MS  = 400;   // capture window — shorter than P300, so faster RSVP
export const N170_WIN_START = 120;   // ms — N170 onset
export const N170_WIN_END   = 220;   // ms — N170 offset
export const VPP_WIN_START  = 140;   // ms — VPP onset
export const VPP_WIN_END    = 220;   // ms — VPP offset

export const ARTIFACT_THRESH_UV = 100;  // reject epochs with |signal| > this
export const RECOMMENDED_RSVP_HZ = 10; // can go faster than P300 (was 8)

// ── EEG ring buffer ──────────────────────────────────────────────────────────
export class EEGBuffer {
  constructor(maxDurationMs = 3000) {
    this.samples = [];
    this.maxDuration = maxDurationMs;
  }

  push(packet) {
    const { data, startTime, sampleRate } = packet;
    const dt = 1000 / sampleRate;
    for (let i = 0; i < data.length; i++) {
      this.samples.push({
        t: startTime + i * dt,
        ch: new Float32Array(data[i]),
      });
    }
    const cutoff = Date.now() - this.maxDuration;
    let trim = 0;
    while (trim < this.samples.length && this.samples[trim].t < cutoff) trim++;
    if (trim > 0) this.samples.splice(0, trim);
  }

  getSamplesInWindow(startMs, endMs) {
    return this.samples.filter(s => s.t >= startMs && s.t <= endMs);
  }

  get length() { return this.samples.length; }
}

// ── Epoch extraction ─────────────────────────────────────────────────────────
/**
 * Extract and baseline-correct an epoch around a stimulus onset.
 * Returns null if insufficient data or artifact detected.
 */
export function extractEpoch(buffer, stimulusTs) {
  const windowStart = stimulusTs - EPOCH_PRE_MS;
  const windowEnd   = stimulusTs + EPOCH_POST_MS;
  const samples     = buffer.getSamplesInWindow(windowStart, windowEnd);

  const expectedSamples = Math.round((EPOCH_PRE_MS + EPOCH_POST_MS) / 1000 * SAMPLE_RATE);
  if (samples.length < expectedSamples * 0.75) return null;

  const baseline = samples.filter(s => s.t < stimulusTs);
  if (baseline.length < 5) return null;

  // Baseline means per channel
  const baselineMeans = new Float32Array(N_CHANNELS);
  for (const s of baseline) {
    for (let c = 0; c < N_CHANNELS; c++) baselineMeans[c] += s.ch[c];
  }
  for (let c = 0; c < N_CHANNELS; c++) baselineMeans[c] /= baseline.length;

  // Baseline-corrected epoch: time relative to stimulus
  const epoch = samples.map(s => ({
    t: s.t - stimulusTs,
    ch: s.ch.map((v, c) => v - baselineMeans[c]),
  }));

  // Artifact rejection across all channels
  for (const s of epoch) {
    for (let c = 0; c < N_CHANNELS; c++) {
      if (Math.abs(s.ch[c]) > ARTIFACT_THRESH_UV) return null;
    }
  }

  return epoch;
}

// ── N170 + VPP scoring ───────────────────────────────────────────────────────
/**
 * Score an epoch for N170 (negative at PO3/PO4) and VPP (positive at C3/C4).
 *
 * N170 score: the most negative peak in the 120–220ms window on PO3/PO4.
 *             Faces produce a stronger (more negative) N170.
 *
 * VPP score:  the most positive peak in the 140–220ms window on C3/C4.
 *             Faces produce a stronger (more positive) VPP.
 *
 * Combined: both components firing together indicates face/body detection.
 *
 * @param {Array}  epoch      - baseline-corrected epoch from extractEpoch()
 * @param {number} n170Thresh - negative µV threshold (e.g. -2.5)
 * @param {number} vppThresh  - positive µV threshold (e.g. 1.5)
 * @returns {{ n170: number, vpp: number, combined: number, isHit: boolean }}
 */
export function scoreEpoch(epoch, n170Thresh = -2.5, vppThresh = 1.5) {
  if (!epoch) return { n170: 0, vpp: 0, combined: 0, score: 0, isHit: false };

  // N170: most negative value in window across PO3+PO4
  const n170Samples = epoch.filter(s => s.t >= N170_WIN_START && s.t <= N170_WIN_END);
  let n170Peak = 0;  // starts at 0; faces make this go negative
  for (const s of n170Samples) {
    for (const c of N170_CHANNELS) {
      if (s.ch[c] < n170Peak) n170Peak = s.ch[c];
    }
  }

  // VPP: most positive value in window across C3+C4
  const vppSamples = epoch.filter(s => s.t >= VPP_WIN_START && s.t <= VPP_WIN_END);
  let vppPeak = 0;   // starts at 0; faces make this go positive
  for (const s of vppSamples) {
    for (const c of VPP_CHANNELS) {
      if (s.ch[c] > vppPeak) vppPeak = s.ch[c];
    }
  }

  // Combined score: N170 negativity (flipped to positive) + VPP positivity
  // Normalised 0–1 against expected thresholds
  const n170Score = Math.max(0, -n170Peak / Math.abs(n170Thresh));
  const vppScore  = Math.max(0,  vppPeak  / vppThresh);
  const combined  = (n170Score + vppScore) / 2;
  const score     = Math.min(1, combined);

  // Hit if N170 crosses threshold AND VPP is at least partially present
  const isHit = n170Peak < n170Thresh && vppPeak > (vppThresh * 0.5);

  return {
    n170: Math.round(n170Peak * 100) / 100,      // µV (negative for faces)
    vpp:  Math.round(vppPeak  * 100) / 100,      // µV (positive for faces)
    score: Math.round(score * 1000) / 1000,
    isHit,
  };
}

// ── Calibrator ───────────────────────────────────────────────────────────────
/**
 * Collect face and non-face epochs, compute grand-average ERP,
 * and return recommended thresholds for N170 and VPP.
 */
export class Calibrator {
  constructor() {
    this.faceEpochs    = [];
    this.nonFaceEpochs = [];
  }

  addEpoch(epoch, isFace) {
    if (!epoch) return;
    if (isFace) this.faceEpochs.push(epoch);
    else        this.nonFaceEpochs.push(epoch);
  }

  get faceCount()    { return this.faceEpochs.length; }
  get nonFaceCount() { return this.nonFaceEpochs.length; }
  get isReady()      { return this.faceEpochs.length >= 10 && this.nonFaceEpochs.length >= 30; }

  compute() {
    if (!this.isReady) return null;

    const avgEpoch = (epochs) => {
      const template = epochs[0].map(s => ({ t: s.t, ch: new Float32Array(N_CHANNELS) }));
      for (const ep of epochs) {
        for (let i = 0; i < Math.min(template.length, ep.length); i++) {
          for (let c = 0; c < N_CHANNELS; c++) {
            template[i].ch[c] += ep[i].ch[c] / epochs.length;
          }
        }
      }
      return template;
    };

    const faceAvg    = avgEpoch(this.faceEpochs);
    const nonFaceAvg = avgEpoch(this.nonFaceEpochs);

    // N170: most negative difference in window on PO3/PO4
    let n170DiffPeak = 0;
    let n170PeakMs   = 0;
    for (const s of faceAvg) {
      if (s.t < N170_WIN_START || s.t > N170_WIN_END) continue;
      const faceIdx   = faceAvg.findIndex(x => x.t === s.t);
      const nonFaceS  = nonFaceAvg[faceIdx];
      if (!nonFaceS) continue;
      for (const c of N170_CHANNELS) {
        const diff = s.ch[c] - (nonFaceS?.ch[c] || 0);
        if (diff < n170DiffPeak) { n170DiffPeak = diff; n170PeakMs = s.t; }
      }
    }

    // VPP: most positive difference in window on C3/C4
    let vppDiffPeak = 0;
    let vppPeakMs   = 0;
    for (const s of faceAvg) {
      if (s.t < VPP_WIN_START || s.t > VPP_WIN_END) continue;
      const faceIdx   = faceAvg.findIndex(x => x.t === s.t);
      const nonFaceS  = nonFaceAvg[faceIdx];
      if (!nonFaceS) continue;
      for (const c of VPP_CHANNELS) {
        const diff = s.ch[c] - (nonFaceS?.ch[c] || 0);
        if (diff > vppDiffPeak) { vppDiffPeak = diff; vppPeakMs = s.t; }
      }
    }

    // Thresholds: 60% of peak difference (more conservative than P300's 50%
    // because N170 single-trial SNR is lower)
    const n170Thresh = Math.min(-1.0, n170DiffPeak * 0.6);
    const vppThresh  = Math.max( 0.8,  vppDiffPeak  * 0.6);

    return {
      n170Thresh:    Math.round(n170Thresh * 100) / 100,
      vppThresh:     Math.round(vppThresh  * 100) / 100,
      n170PeakMs:    Math.round(n170PeakMs),
      vppPeakMs:     Math.round(vppPeakMs),
      n170Amplitude: Math.round(n170DiffPeak * 100) / 100,
      vppAmplitude:  Math.round(vppDiffPeak  * 100) / 100,
      faceAvg,
      nonFaceAvg,
      nFaces:    this.faceEpochs.length,
      nNonFaces: this.nonFaceEpochs.length,
    };
  }

  reset() {
    this.faceEpochs    = [];
    this.nonFaceEpochs = [];
  }
}
