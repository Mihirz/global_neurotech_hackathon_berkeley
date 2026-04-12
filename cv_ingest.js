// Continuous CV ingestion: samples the drone-video element and POSTs each
// frame to the detector sidecar. Also forwards P300 hits for fusion.

const DETECTOR_URL = window.DETECTOR_URL || 'http://localhost:8000';
const INGEST_HZ    = 2;   // frames per second sent to the detector
const JPEG_QUALITY = 0.6;
const MAX_WIDTH    = 960; // keep small aerial people visible for YOLO

const cvState = {
  running:   false,
  interval:  null,
  frameId:   0,
  inFlight:  0,
  maxInFlight: 2,
  lastResult: null,
  onFrame:    null, // optional callback(result)
};

function sampleFrame(videoEl, canvasEl) {
  if (!videoEl || videoEl.readyState < 2 || videoEl.videoWidth === 0) return null;
  const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
  const scale = Math.min(1, MAX_WIDTH / vw);
  const w = Math.round(vw * scale), h = Math.round(vh * scale);
  canvasEl.width = w; canvasEl.height = h;
  const ctx = canvasEl.getContext('2d');
  ctx.drawImage(videoEl, 0, 0, w, h);
  return canvasEl.toDataURL('image/jpeg', JPEG_QUALITY);
}

async function ingestOnce(getContext) {
  if (cvState.inFlight >= cvState.maxInFlight) return;
  const { videoEl, canvasEl, gps, recentEegHit } = getContext();
  const dataUrl = sampleFrame(videoEl, canvasEl);
  if (!dataUrl) return;

  const frame_id = cvState.frameId++;
  const body = {
    frame_id,
    image: dataUrl.split(',', 2)[1],
    ts: Date.now(),
    gps: gps || null,
    eeg_flagged:   !!recentEegHit,
    eeg_amplitude: recentEegHit?.amplitude ?? null,
    eeg_score:     recentEegHit?.score ?? null,
  };

  cvState.inFlight++;
  try {
    const res = await fetch(`${DETECTOR_URL}/ingest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (res.ok) {
      cvState.lastResult = await res.json();
      cvState.onFrame?.(cvState.lastResult);
    }
  } catch (err) {
    // Silently swallow; insights UI shows "detector offline" separately.
  } finally {
    cvState.inFlight--;
  }
}

export function startCvIngest({ videoElId = 'drone-video', canvasElId = 'cv-canvas', getGps, onFrame } = {}) {
  if (cvState.running) return;
  cvState.running = true;
  cvState.onFrame = onFrame || null;

  // Hidden canvas for sampling (separate from RSVP frame-canvas).
  let canvasEl = document.getElementById(canvasElId);
  if (!canvasEl) {
    canvasEl = document.createElement('canvas');
    canvasEl.id = canvasElId;
    canvasEl.style.display = 'none';
    document.body.appendChild(canvasEl);
  }
  const videoEl = document.getElementById(videoElId);

  const getContext = () => ({
    videoEl,
    canvasEl,
    gps: getGps ? getGps() : null,
    recentEegHit: cvState.pendingEegHit,
  });

  cvState.interval = setInterval(() => {
    ingestOnce(getContext);
    // Clear the pending EEG tag after we've attached it to a frame.
    cvState.pendingEegHit = null;
  }, 1000 / INGEST_HZ);
}

export function stopCvIngest() {
  cvState.running = false;
  if (cvState.interval) clearInterval(cvState.interval);
  cvState.interval = null;
}

export function notifyEegHit(det) {
  // Mark the next ingested frame as EEG-flagged.
  cvState.pendingEegHit = {
    frameId:   det.frameId,
    amplitude: det.amplitude,
    score:     det.score,
    ts:        det.ts,
  };
  // Also POST to the detector so it can log standalone EEG hits even when the
  // CV cadence hasn't captured a frame at that exact moment.
  fetch(`${DETECTOR_URL}/eeg-hit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      frame_id:  det.frameId,
      ts:        det.ts,
      amplitude: det.amplitude,
      score:     det.score,
      gps:       det.gps || null,
    }),
  }).catch(() => {});
}

export async function fetchInsights() {
  try {
    const res = await fetch(`${DETECTOR_URL}/insights`);
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function resetDetector() {
  try {
    await fetch(`${DETECTOR_URL}/reset`, { method: 'POST' });
  } catch {}
  cvState.frameId = 0;
}

export function getDetectorUrl() { return DETECTOR_URL; }
