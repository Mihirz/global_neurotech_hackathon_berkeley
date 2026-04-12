/**
 * eeg_synthetic.js — Synthetic EEG generator for N170/VPP classifier training
 *
 * Generates realistic Emotiv Insight-format EEG (128Hz, 5ch) calibrated to
 * published N170/VPP literature values, with inter-trial variability and
 * controller-use artifacts (EMG, eye movement, slow drift, 1/f background).
 *
 * Emotiv Insight channel map:
 *   Index  Label   Position          N170/VPP role
 *     0    AF3     Left frontal      minimal N170 spread, frontal beta
 *     1    T7      Left temporal     PRIMARY N170 (posterior temporal)
 *     2    Pz      Parietal midline  PRIMARY VPP
 *     3    T8      Right temporal    PRIMARY N170
 *     4    AF4     Right frontal     minimal N170 spread, frontal beta
 *
 * N170 reference values (face vs. non-face):
 *   Peak amplitude : −3 to −6 µV at T7/T8 (single trial std ~2 µV)
 *   Peak latency   : 160–200 ms (mean 170 ms, std ~15 ms)
 *   Distribution   : bilateral, slight left dominance for upright faces
 *
 * VPP reference values:
 *   Peak amplitude : +2 to +5 µV at Pz (single trial std ~1 µV)
 *   Peak latency   : 150–200 ms (co-occurs with N170, mean ~175 ms)
 *
 * Background SNR reflects real single-trial conditions:
 *   Background RMS : 8–14 µV (alpha + pink noise dominant)
 *   Single-trial SNR: ~0.3–0.5 (realistic; averaging improves this)
 *
 * Usage:
 *   import { generateTrainingDataset, SyntheticEEGStream } from './eeg_synthetic.js';
 *
 *   // Offline labeled training set (200 face + 200 non-face trials):
 *   const dataset = generateTrainingDataset(200, { controllerActivity: 0.6 });
 *
 *   // Streaming drop-in for server.js demo mode:
 *   const stream = new SyntheticEEGStream({ onPacket, onStatus });
 *   stream.start();
 *   stream.notifyFrameOnset(ts, isFace);   // call on each RSVP frame
 */

// ── Constants ──────────────────────────────────────────────────────────────────
export const SAMPLE_RATE = 128;
export const N_CH        = 5;
export const DT          = 1000 / SAMPLE_RATE;  // ~7.8125 ms

// Channel indices (matches erp.js)
const CH = { AF3: 0, T7: 1, Pz: 2, T8: 3, AF4: 4 };

// Epoch windows (ms) — must match erp.js EPOCH_PRE_MS / EPOCH_POST_MS
const PRE_MS  = 100;
const POST_MS = 400;

// ── Statistical helpers ────────────────────────────────────────────────────────

/** Box-Muller standard normal sample */
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** Clamp a value to [lo, hi] */
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }

// ── Noise model ────────────────────────────────────────────────────────────────

/**
 * Create per-channel pink (1/f) noise state.
 * Uses a 3-pole Voss-McCartney IIR approximation.
 * Gives realistic 1/f spectrum up to ~50 Hz.
 */
function createPinkState() {
  return { b0: 0, b1: 0, b2: 0 };
}

function pinkSample(s, sigma) {
  const white = randn() * sigma;
  s.b0 = 0.99886 * s.b0 + white * 0.0555179;
  s.b1 = 0.99332 * s.b1 + white * 0.0750759;
  s.b2 = 0.96900 * s.b2 + white * 0.1538520;
  return (s.b0 + s.b1 + s.b2 + white * 0.5362) * 0.11;
}

// ── ERP templates ──────────────────────────────────────────────────────────────

/** Gaussian-shaped component: value at time t given peak, width, amplitude */
function gauss(t, peak, width, amp) {
  return amp * Math.exp(-((t - peak) ** 2) / (2 * width * width));
}

/**
 * Spatial topography weights for each ERP component.
 * Values are relative amplitudes at [AF3, T7, Pz, T8, AF4].
 *
 * Based on published N170 scalp maps (Bentin et al., 1996; Eimer, 2000)
 * adapted to the 5-electrode Insight layout.
 */
const TOPO = {
  // Early visual components — generic stimulus response
  P1:   [0.20, 0.65, 0.80, 0.65, 0.20],   // 80–120ms, occipito-parietal
  N1:   [0.20, 0.75, 0.35, 0.75, 0.20],   // 100–150ms, temporal

  // Face-selective components
  N170: [0.15, 1.00, 0.05, 0.88, 0.15],   // T7 dominant, slight left bias
  VPP:  [0.12, -0.08, 1.00, -0.08, 0.12], // Pz dominant, antiphase at T7/T8

  // Non-face object P2 component
  P2obj:[0.20, 0.45, 0.65, 0.45, 0.20],   // smaller, diffuse

  // Late cognitive component (target processing)
  P3:   [0.25, 0.30, 0.90, 0.30, 0.25],   // Pz-maximal
};

/**
 * Compute instantaneous ERP voltage at time dt (ms post-stimulus).
 * Returns Float32Array[5] of µV contributions.
 *
 * @param {number} dt           ms post-stimulus
 * @param {boolean} isFace      whether stimulus contains a person
 * @param {object}  params      per-trial random parameters
 */
function computeERP(dt, isFace, params) {
  const ch = new Float32Array(N_CH);
  if (dt < 0 || dt > POST_MS) return ch;

  // ── P1 (80–120ms): early visual, both conditions ─────────────────────────
  const p1 = gauss(dt, 100, 14, 1.6 + randn() * 0.3);
  for (let c = 0; c < N_CH; c++) ch[c] += TOPO.P1[c] * p1;

  // ── N1 (100–150ms): visual N1/C1, both conditions ────────────────────────
  const n1 = gauss(dt, 130, 17, -(1.8 + randn() * 0.4));
  for (let c = 0; c < N_CH; c++) ch[c] += TOPO.N1[c] * n1;

  if (isFace) {
    // ── N170 (120–220ms): face-specific negative temporal component ─────────
    // Single-trial amplitude: mean −4.5µV, std 1.5µV; latency: mean 170ms, std 15ms
    const n170 = gauss(dt, params.n170Lat, 24, params.n170Amp);
    for (let c = 0; c < N_CH; c++) ch[c] += TOPO.N170[c] * n170;

    // ── VPP (140–220ms): vertex positive potential, co-occurs with N170 ─────
    // Amplitude: mean +3.2µV, std 1.0µV; latency slightly later than N170
    const vpp = gauss(dt, params.vppLat, 20, params.vppAmp);
    for (let c = 0; c < N_CH; c++) ch[c] += TOPO.VPP[c] * vpp;

    // ── Late positivity (280–400ms): task-relevant recognition ───────────────
    const p3 = gauss(dt, 320, 55, params.p3Amp);
    for (let c = 0; c < N_CH; c++) ch[c] += TOPO.P3[c] * p3;

  } else {
    // ── P2 for non-face objects (180–240ms): present but no N170/VPP ────────
    const p2 = gauss(dt, 200, 28, 0.9 + randn() * 0.3);
    for (let c = 0; c < N_CH; c++) ch[c] += TOPO.P2obj[c] * p2;
  }

  return ch;
}

/**
 * Sample per-trial random ERP parameters.
 * Draws from distributions that match published N170/VPP variability.
 */
function sampleERPParams(isFace) {
  if (!isFace) return {};
  return {
    // N170: slightly left-lateralized mean, std 1.5µV
    n170Amp: clamp(randn() * 1.5 - 4.5, -8.0, -1.5),
    // N170 latency: 170ms mean, std 15ms (range 130–220ms)
    n170Lat: clamp(170 + randn() * 15, 130, 220),
    // VPP: mean +3.2µV, std 1.0µV
    vppAmp:  clamp(randn() * 1.0 + 3.2,  1.0,  6.0),
    // VPP latency: slightly later than N170
    vppLat:  clamp(175 + randn() * 12, 140, 220),
    // Late P3 amplitude (variable, task-dependent)
    p3Amp:   clamp(randn() * 1.0 + 2.5,  0.5,  5.0),
  };
}

// ── Artifact generators ────────────────────────────────────────────────────────

/**
 * Create an EMG burst state (controller button press / grip tightening).
 * Returns null with probability (1 - controllerActivity * 0.35).
 */
function maybeEMGBurst(epochDurationMs, controllerActivity) {
  if (Math.random() > controllerActivity * 0.35) return null;
  return {
    onset:    Math.random() * epochDurationMs,
    duration: 60 + Math.random() * 140,       // 60–200ms
    amp:      5 + Math.random() * 15,          // 5–20µV RMS
    freq:     50 + Math.random() * 50,         // 50–100Hz carrier
    // EMG is strongest at temporal (jaw/neck) and frontal (brow) muscles
    weights:  [0.55, 1.0, 0.08, 1.0, 0.55],
  };
}

/**
 * Create an eye blink state.
 * ~0.2 blinks/second at rest; controller use reduces this (gaze-on-screen).
 */
function maybeBlink(epochDurationMs) {
  if (Math.random() > 0.10) return null;
  return {
    onset: Math.random() * epochDurationMs,
    amp:   40 + Math.random() * 60,            // 40–100µV
    // Blink propagates strongly to frontal, weakly to posterior channels
    weights: [1.00, 0.22, 0.04, 0.22, 0.90],
  };
}

/** Compute blink voltage at dt ms after blink onset */
function blinkProfile(dt) {
  if (dt < 0 || dt > 450) return 0;
  if (dt < 100) return dt / 100;                  // fast rise ~100ms
  return Math.exp(-(dt - 100) / 180);             // slow exponential decay
}

// ── Single-epoch generator ────────────────────────────────────────────────────

/**
 * Generate one baseline-corrected epoch in erp.js format.
 *
 * Returns Array<{t: number, ch: Float32Array}> where:
 *   t  = ms relative to stimulus onset (negative = pre-stimulus)
 *   ch = [AF3, T7, Pz, T8, AF4] in µV
 *
 * @param {boolean} isFace         - true if stimulus contains a person
 * @param {object}  opts
 * @param {number}  opts.controllerActivity  - 0 (none) to 1 (heavy use)
 * @param {number}  opts.alphaAmplitude      - override alpha amplitude (µV)
 */
export function generateEpoch(isFace, opts = {}) {
  const {
    controllerActivity = 0.5,
    alphaAmplitude     = 8.0,   // µV; controller use suppresses alpha slightly
  } = opts;

  const totalMs  = PRE_MS + POST_MS;
  const nSamples = Math.round(totalMs / DT);

  // Per-trial ERP parameters
  const erpParams = sampleERPParams(isFace);

  // Random oscillation phases (not phase-locked to stimulus — realistic)
  const alphaPhase = Math.random() * 2 * Math.PI;
  const betaPhase  = Math.random() * 2 * Math.PI;
  const thetaPhase = Math.random() * 2 * Math.PI;
  // Slow drift: 0.05–0.15 Hz, 2–8µV
  const driftFreq  = 0.05 + Math.random() * 0.10;
  const driftPhase = Math.random() * 2 * Math.PI;
  const driftAmp   = 2 + Math.random() * 6;

  // Per-channel 1/f noise state
  const pink = Array.from({ length: N_CH }, createPinkState);

  // Artifacts — occurrence depends on controller activity
  const emg   = maybeEMGBurst(totalMs, controllerActivity);
  const blink = maybeBlink(totalMs);

  // Controller use slightly suppresses alpha (visual engagement) and boosts beta
  const alphaMod = 1.0 - controllerActivity * 0.25;
  const betaMod  = 1.0 + controllerActivity * 0.40;

  // Spatially-weighted oscillation amplitudes per channel
  // Alpha: occipital/parietal dominant → Pz strongest
  // Beta:  frontal/motor dominant → AF3/AF4 strongest
  // Theta: widespread, slightly frontal
  const alphaW = [0.22, 0.42, 1.00, 0.42, 0.22].map(w => w * alphaAmplitude * alphaMod);
  const betaW  = [1.00, 0.45, 0.28, 0.45, 1.00].map(w => w * 3.0 * betaMod);
  const thetaW = [0.55, 0.55, 0.65, 0.55, 0.55].map(w => w * 5.0);

  // Accumulate raw samples (absolute-time index, then baseline-correct)
  const rawEpoch = [];

  for (let i = 0; i < nSamples; i++) {
    const tAbs = i * DT;          // ms from epoch start (0 = start of pre-stimulus)
    const t    = tAbs - PRE_MS;   // ms relative to stimulus onset
    const tSec = tAbs / 1000;

    const ch = new Float32Array(N_CH);

    for (let c = 0; c < N_CH; c++) {
      // Oscillatory background
      ch[c] += alphaW[c] * Math.sin(2 * Math.PI * 10.0 * tSec + alphaPhase);
      ch[c] += betaW[c]  * Math.sin(2 * Math.PI * 20.0 * tSec + betaPhase);
      ch[c] += thetaW[c] * Math.sin(2 * Math.PI * 6.0  * tSec + thetaPhase);

      // Pink (1/f) noise — primary source of broadband background
      ch[c] += pinkSample(pink[c], 3.0);

      // White noise floor
      ch[c] += randn() * 0.8;

      // Slow drift (movement, electrode drift)
      ch[c] += driftAmp * Math.sin(2 * Math.PI * driftFreq * tSec + driftPhase);
    }

    // EMG artifact
    if (emg) {
      const dt = tAbs - emg.onset;
      if (dt >= 0 && dt <= emg.duration) {
        const envelope = Math.sin(Math.PI * dt / emg.duration); // smooth onset/offset
        for (let c = 0; c < N_CH; c++) {
          ch[c] += emg.weights[c] * emg.amp * envelope *
                   Math.sin(2 * Math.PI * emg.freq * tSec) * randn() * 0.7;
        }
      }
    }

    // Eye blink
    if (blink) {
      const blinkV = blink.amp * blinkProfile(tAbs - blink.onset);
      for (let c = 0; c < N_CH; c++) ch[c] += blink.weights[c] * blinkV;
    }

    // ERP (post-stimulus only)
    if (t >= 0) {
      const erpContrib = computeERP(t, isFace, erpParams);
      for (let c = 0; c < N_CH; c++) ch[c] += erpContrib[c];
    }

    rawEpoch.push({ tAbs, t, ch });
  }

  // ── Baseline correction (mean of pre-stimulus window) ─────────────────────
  const baseline = rawEpoch.filter(s => s.t < 0);
  const means = new Float32Array(N_CH);
  for (const s of baseline) for (let c = 0; c < N_CH; c++) means[c] += s.ch[c];
  for (let c = 0; c < N_CH; c++) means[c] /= baseline.length;

  const epoch = rawEpoch.map(s => ({
    t:  s.t,
    ch: s.ch.map((v, c) => v - means[c]),
  }));

  return epoch;
}

// ── Training dataset generator ─────────────────────────────────────────────────

/**
 * Generate a balanced, labeled training dataset for N170/VPP classification.
 *
 * Returns an array of trial objects compatible with erp.js extractEpoch output.
 * Trials are shuffled to prevent order bias in training.
 *
 * @param {number} nTrialsPerClass  - number of face AND non-face trials each
 * @param {object} opts
 * @param {number} opts.controllerActivity  - 0–1, controls EMG/artifact rate
 * @param {boolean} opts.keepArtifacted     - include artifact-contaminated epochs
 * @param {boolean} opts.includeFeatures    - pre-compute N170/VPP feature values
 *
 * @returns {Array<{
 *   label: 'face'|'nonface',
 *   isFace: boolean,
 *   isArtifacted: boolean,
 *   epoch: Array<{t:number, ch:Float32Array}>,
 *   features?: {n170_T7, n170_T8, n170_mean, vpp_Pz, combined, score}
 * }>}
 */
export function generateTrainingDataset(nTrialsPerClass = 200, opts = {}) {
  const {
    controllerActivity = 0.5,
    keepArtifacted     = false,
    includeFeatures    = true,
  } = opts;

  const trials = [];

  for (let i = 0; i < nTrialsPerClass; i++) {
    for (const isFace of [true, false]) {
      const epoch = generateEpoch(isFace, { controllerActivity });

      // Artifact check (matches erp.js ARTIFACT_THRESH_UV = 100)
      const maxAmp = Math.max(...epoch.flatMap(s => Array.from(s.ch).map(Math.abs)));
      const isArtifacted = maxAmp > 100;

      if (!keepArtifacted && isArtifacted) continue;

      const trial = {
        label:        isFace ? 'face' : 'nonface',
        isFace,
        isArtifacted,
        epoch,
      };

      if (includeFeatures) {
        trial.features = extractFeatures(epoch);
      }

      trials.push(trial);
    }
  }

  // Fisher-Yates shuffle
  for (let i = trials.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [trials[i], trials[j]] = [trials[j], trials[i]];
  }

  return trials;
}

/**
 * Pre-compute N170/VPP features from a baseline-corrected epoch.
 * Uses the same time windows as erp.js scoreEpoch().
 *
 * @param {Array<{t:number, ch:Float32Array}>} epoch
 * @returns {{ n170_T7, n170_T8, n170_mean, vpp_Pz, combined, score }}
 */
export function extractFeatures(epoch) {
  // N170 window: 120–220ms (erp.js N170_WIN_START / N170_WIN_END)
  const n170Win = epoch.filter(s => s.t >= 120 && s.t <= 220);
  // VPP window: 140–220ms (erp.js VPP_WIN_START / VPP_WIN_END)
  const vppWin  = epoch.filter(s => s.t >= 140 && s.t <= 220);

  let n170_T7 = 0, n170_T8 = 0;
  for (const s of n170Win) {
    if (s.ch[CH.T7] < n170_T7) n170_T7 = s.ch[CH.T7];
    if (s.ch[CH.T8] < n170_T8) n170_T8 = s.ch[CH.T8];
  }
  const n170_mean = (n170_T7 + n170_T8) / 2;

  let vpp_Pz = 0;
  for (const s of vppWin) {
    if (s.ch[CH.Pz] > vpp_Pz) vpp_Pz = s.ch[CH.Pz];
  }

  // Normalised 0–1 score matching erp.js scoreEpoch() formula
  const n170Score = Math.max(0, -n170_mean / 2.5);
  const vppScore  = Math.max(0,  vpp_Pz   / 1.5);
  const combined  = (n170Score + vppScore) / 2;
  const score     = Math.min(1, combined);

  return {
    n170_T7:   Math.round(n170_T7   * 100) / 100,
    n170_T8:   Math.round(n170_T8   * 100) / 100,
    n170_mean: Math.round(n170_mean * 100) / 100,
    vpp_Pz:    Math.round(vpp_Pz    * 100) / 100,
    combined:  Math.round(combined  * 1000) / 1000,
    score:     Math.round(score     * 1000) / 1000,
  };
}

/**
 * Summarise dataset statistics — useful to verify the generator is calibrated.
 *
 * @param {Array} trials  - output of generateTrainingDataset()
 * @returns {object}  per-class means, stds, and separability
 */
export function datasetStats(trials) {
  const face    = trials.filter(t => t.isFace    && t.features);
  const nonFace = trials.filter(t => !t.isFace   && t.features);

  const mean = (arr, key) => arr.reduce((s, t) => s + t.features[key], 0) / arr.length;
  const std  = (arr, key, m) => {
    const v = arr.reduce((s, t) => s + (t.features[key] - m) ** 2, 0) / arr.length;
    return Math.sqrt(v);
  };

  const stats = {};
  for (const key of ['n170_T7', 'n170_T8', 'n170_mean', 'vpp_Pz', 'score']) {
    const fm = mean(face,    key);
    const nm = mean(nonFace, key);
    const fs = std(face,    key, fm);
    const ns = std(nonFace, key, nm);
    const pooledSd = Math.sqrt((fs ** 2 + ns ** 2) / 2);
    stats[key] = {
      face_mean:    Math.round(fm * 100) / 100,
      nonface_mean: Math.round(nm * 100) / 100,
      face_std:     Math.round(fs * 100) / 100,
      nonface_std:  Math.round(ns * 100) / 100,
      // Cohen's d: effect size (>0.8 is large; real N170 typically 0.5–1.2)
      cohens_d:     Math.round(Math.abs(fm - nm) / (pooledSd || 1) * 100) / 100,
    };
  }

  return {
    nFace:       face.length,
    nNonFace:    nonFace.length,
    nArtifacted: trials.filter(t => t.isArtifacted).length,
    features:    stats,
  };
}

// ── Streaming EEG (server.js drop-in) ─────────────────────────────────────────

/**
 * SyntheticEEGStream — replaces server.js startSimulatedEEG().
 *
 * Produces continuous 128Hz Insight-format packets with realistic background EEG.
 * When notifyFrameOnset(ts, isFace) is called, injects the appropriate
 * N170/VPP (face) or P1/N1/P2 (non-face) ERP at the correct latency.
 *
 * Controller-use artifacts are always active at the configured level.
 *
 * @param {object} opts
 * @param {function} opts.onPacket            - callback({data, startTime, sampleRate, channels})
 * @param {function} opts.onStatus            - status string callback
 * @param {number}   opts.controllerActivity  - 0–1 (default 0.5)
 */
export class SyntheticEEGStream {
  constructor({ onPacket, onStatus, controllerActivity = 0.5 }) {
    this.onPacket           = onPacket;
    this.onStatus           = onStatus || (() => {});
    this.controllerActivity = controllerActivity;

    this._interval   = null;
    this._t          = Date.now();       // current simulation time (ms)
    this._pink       = Array.from({ length: N_CH }, createPinkState);
    this._alphaPhase = Array.from({ length: N_CH }, () => Math.random() * 2 * Math.PI);
    this._betaPhase  = Array.from({ length: N_CH }, () => Math.random() * 2 * Math.PI);
    this._thetaPhase = Array.from({ length: N_CH }, () => Math.random() * 2 * Math.PI);

    // Active ERP events: [{onset, isFace, params}]
    this._pendingERPs = [];

    // Active artifact events
    this._activeEMG   = null;
    this._activeBlink = null;
    this._emgCooldown = 0;
    this._blinkCooldown = 0;
  }

  /**
   * Notify the stream that an RSVP frame was shown at time `ts`.
   * If isFace=true, an N170+VPP will be injected at ts+170ms.
   * If isFace=false, only P1/N1/P2 (present for all stimuli) is injected.
   */
  notifyFrameOnset(ts, isFace = false) {
    this._pendingERPs.push({
      onset:  ts,
      isFace,
      params: sampleERPParams(isFace),
    });
    // Expire old ERPs (> 600ms old)
    this._pendingERPs = this._pendingERPs.filter(e => e.onset > Date.now() - 650);
  }

  /** Generate one raw sample at absolute time tMs. Returns Float32Array[5]. */
  _sample(tMs) {
    const tSec = tMs / 1000;
    const ch   = new Float32Array(N_CH);

    const alphaMod = 1.0 - this.controllerActivity * 0.25;
    const betaMod  = 1.0 + this.controllerActivity * 0.40;

    for (let c = 0; c < N_CH; c++) {
      // Advance oscillator phases continuously (avoids phase discontinuities)
      this._alphaPhase[c] += (2 * Math.PI * 10) / SAMPLE_RATE;
      this._betaPhase[c]  += (2 * Math.PI * 20) / SAMPLE_RATE;
      this._thetaPhase[c] += (2 * Math.PI * 6)  / SAMPLE_RATE;

      const alphaW = [0.22, 0.42, 1.00, 0.42, 0.22][c] * 8.0 * alphaMod;
      const betaW  = [1.00, 0.45, 0.28, 0.45, 1.00][c] * 3.0 * betaMod;
      const thetaW = [0.55, 0.55, 0.65, 0.55, 0.55][c] * 5.0;

      ch[c] += alphaW * Math.sin(this._alphaPhase[c]);
      ch[c] += betaW  * Math.sin(this._betaPhase[c]);
      ch[c] += thetaW * Math.sin(this._thetaPhase[c]);
      ch[c] += pinkSample(this._pink[c], 3.0);
      ch[c] += randn() * 0.8;
    }

    // Spontaneous EMG burst (controller grip)
    if (!this._activeEMG && tMs > this._emgCooldown) {
      if (Math.random() < this.controllerActivity * 0.002) {  // ~0.5/s at full activity
        this._activeEMG = {
          start:    tMs,
          duration: 60 + Math.random() * 140,
          amp:      5  + Math.random() * 15,
          freq:     50 + Math.random() * 50,
          weights:  [0.55, 1.0, 0.08, 1.0, 0.55],
        };
      }
    }
    if (this._activeEMG) {
      const dt = tMs - this._activeEMG.start;
      if (dt <= this._activeEMG.duration) {
        const env = Math.sin(Math.PI * dt / this._activeEMG.duration);
        for (let c = 0; c < N_CH; c++) {
          ch[c] += this._activeEMG.weights[c] * this._activeEMG.amp * env *
                   Math.sin(2 * Math.PI * this._activeEMG.freq * tSec) * (randn() * 0.5 + 0.5);
        }
      } else {
        this._emgCooldown = tMs + 200 + Math.random() * 800;
        this._activeEMG = null;
      }
    }

    // Spontaneous blink
    if (!this._activeBlink && tMs > this._blinkCooldown) {
      if (Math.random() < 0.0003) {  // ~0.04 blinks/s (reduced during task)
        this._activeBlink = {
          start:   tMs,
          amp:     40 + Math.random() * 60,
          weights: [1.00, 0.22, 0.04, 0.22, 0.90],
        };
      }
    }
    if (this._activeBlink) {
      const dt  = tMs - this._activeBlink.start;
      const blinkV = this._activeBlink.amp * blinkProfile(dt);
      if (dt > 450) {
        this._blinkCooldown = tMs + 1000 + Math.random() * 2000;
        this._activeBlink = null;
      } else {
        for (let c = 0; c < N_CH; c++) ch[c] += this._activeBlink.weights[c] * blinkV;
      }
    }

    // ERP contributions from pending frame onsets
    for (const erp of this._pendingERPs) {
      const dt = tMs - erp.onset;
      if (dt < 0 || dt > POST_MS) continue;
      const erpCh = computeERP(dt, erp.isFace, erp.params);
      for (let c = 0; c < N_CH; c++) ch[c] += erpCh[c];
    }

    return ch;
  }

  start() {
    const BATCH_SIZE = 4;
    const packetMs   = (BATCH_SIZE / SAMPLE_RATE) * 1000; // ~31.25ms

    this.onStatus('Synthetic EEG — 128Hz · N170/VPP calibrated · controller artifacts active');

    this._interval = setInterval(() => {
      const packetStart = this._t;
      const data = [];

      for (let i = 0; i < BATCH_SIZE; i++) {
        const sampleT = this._t;
        this._t += 1000 / SAMPLE_RATE;
        data.push(Array.from(this._sample(sampleT)));
      }

      this.onPacket({
        data,
        startTime:  packetStart,
        sampleRate: SAMPLE_RATE,
        channels:   ['AF3', 'T7', 'Pz', 'T8', 'AF4'],
        simulated:  true,
        synthetic:  'n170vpp',
      });
    }, packetMs);
  }

  stop() {
    if (this._interval) {
      clearInterval(this._interval);
      this._interval = null;
    }
  }
}

// ── CLI: generate and print a sample dataset for inspection ───────────────────
// Run directly: `node eeg_synthetic.js`
if (process.argv[1] && process.argv[1].endsWith('eeg_synthetic.js')) {
  const n = parseInt(process.argv[2] || '100', 10);
  console.log(`Generating ${n} face + ${n} non-face trials...`);

  const dataset = generateTrainingDataset(n, { controllerActivity: 0.6 });
  const stats   = datasetStats(dataset);

  console.log('\n── Dataset stats ──────────────────────────────────');
  console.log(`Trials: ${stats.nFace} face, ${stats.nNonFace} non-face`);
  console.log(`Artifacted (>100µV): ${stats.nArtifacted} (${(stats.nArtifacted / dataset.length * 100).toFixed(1)}%)`);
  console.log('\nFeature separability (Cohen\'s d > 0.8 = large effect):');
  for (const [key, v] of Object.entries(stats.features)) {
    const bar = '█'.repeat(Math.round(Math.min(v.cohens_d, 3) * 10));
    console.log(`  ${key.padEnd(12)}: face=${String(v.face_mean).padStart(6)}µV  nonface=${String(v.nonface_mean).padStart(6)}µV  d=${v.cohens_d.toFixed(2)}  ${bar}`);
  }

  // Write JSON for downstream ML use
  const outPath = './training_data.json';
  const fs = await import('fs');
  const out = dataset.map(t => ({
    label:        t.label,
    isFace:       t.isFace,
    isArtifacted: t.isArtifacted,
    features:     t.features,
    // epoch as flat [nSamples × nChannels] array for ML frameworks
    epochMatrix:  t.epoch.map(s => Array.from(s.ch)),
    epochTimes:   t.epoch.map(s => s.t),
  }));
  fs.writeFileSync(outPath, JSON.stringify(out, null, 2));
  console.log(`\nDataset written to ${outPath}`);
}
