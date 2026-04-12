import { EEGBuffer, Calibrator, extractEpoch, scoreEpoch, EPOCH_POST_MS } from './p300.js';

// ─── State ───────────────────────────────────────────────────────────────────
const state = {
  mode: 'idle',         // idle | calibrating | scanning | paused
  p300Threshold: 3.0,   // µV — updated after calibration
  ws: null,
  isDemo: true,
  droneUrl: null,

  // EEG
  eegBuffer: new EEGBuffer(4000),
  calibrator: new Calibrator(),

  // RSVP
  frameQueue: [],
  frameId: 0,
  rsvpInterval: null,
  stimulusLog: [],      // [{frameId, ts, imgSrc, score, isP300, epoch}]
  presentationRateHz: 8,

  // Calibration
  calImages: [],        // {src, isTarget} pre-labelled image set
  calIndex: 0,

  // Results
  flaggedFrames: [],    // frames with detected P300

  // Signal display
  eegPlotBuffer: [],    // last N µV values for waveform display
};

// ─── WebSocket ────────────────────────────────────────────────────────────────
function connectWS() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  state.ws = new WebSocket(`${protocol}://${location.host}`);

  state.ws.onopen = () => {
    log('WebSocket connected');
    setStatusDot('ws-dot', 'green');
  };

  state.ws.onclose = () => {
    setStatusDot('ws-dot', 'red');
    setTimeout(connectWS, 2000);
  };

  state.ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);

    if (msg.type === 'init') {
      state.isDemo = msg.payload.demo;
      state.droneUrl = msg.payload.droneUrl;
      if (state.droneUrl) initDroneFeed(state.droneUrl);
      document.getElementById('demo-badge').style.display = state.isDemo ? 'flex' : 'none';
    }

    if (msg.type === 'crown_status') {
      const s = msg.payload;
      document.getElementById('crown-status').textContent = s.message || s.state;
      const online = s.state === 'online' || s.state === 'demo';
      setStatusDot('eeg-dot', online ? 'green' : (s.state === 'error' ? 'red' : 'amber'));
    }

    if (msg.type === 'eeg_packet') {
      state.eegBuffer.push(msg.payload);
      updateEEGPlot(msg.payload);
    }
  };
}

function wsSend(type, payload = {}) {
  if (state.ws?.readyState === 1) {
    state.ws.send(JSON.stringify({ type, ...payload }));
  }
}

// ─── Drone feed ───────────────────────────────────────────────────────────────
function initDroneFeed(url) {
  const video = document.getElementById('drone-video');
  video.src = url;
  video.play().catch(() => {});
  document.getElementById('no-feed').style.display = 'none';
  video.style.display = 'block';
  log(`Drone feed: ${url}`);
}

// Demo: sample from a static video element or canvas pattern
function sampleDroneFrame() {
  const video = document.getElementById('drone-video');
  const canvas = document.getElementById('frame-canvas');
  const ctx = canvas.getContext('2d');

  if (video.readyState >= 2 && video.videoWidth > 0) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
  } else {
    // Simulated frame: draw thermal-style noise pattern
    canvas.width = 320;
    canvas.height = 240;
    drawSimulatedThermalFrame(ctx, canvas.width, canvas.height, state.frameId);
  }

  return canvas.toDataURL('image/jpeg', 0.75);
}

function drawSimulatedThermalFrame(ctx, w, h, frameId) {
  // Dark thermal background
  ctx.fillStyle = '#0a0f1a';
  ctx.fillRect(0, 0, w, h);

  // Random noise overlay
  const imageData = ctx.getImageData(0, 0, w, h);
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = Math.floor(Math.random() * 30);
    d[i] = v * 0.3; d[i+1] = v * 0.8; d[i+2] = v * 1.2; d[i+3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);

  // Every ~20 frames, draw a "person" heat signature
  const showPerson = (frameId % 20 < 4);
  if (showPerson) {
    const px = 100 + Math.sin(frameId * 0.3) * 60;
    const py = 80 + Math.cos(frameId * 0.2) * 30;
    // Body heat blob
    const grad = ctx.createRadialGradient(px, py+20, 5, px, py+20, 45);
    grad.addColorStop(0, 'rgba(255,200,100,0.9)');
    grad.addColorStop(0.5, 'rgba(255,100,50,0.5)');
    grad.addColorStop(1, 'rgba(255,50,0,0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(px, py+20, 18, 35, 0, 0, Math.PI * 2);
    ctx.fill();
    // Head
    ctx.fillStyle = 'rgba(255,220,120,0.85)';
    ctx.beginPath();
    ctx.arc(px, py, 12, 0, Math.PI * 2);
    ctx.fill();
  }

  // Frame counter
  ctx.fillStyle = 'rgba(100,255,150,0.6)';
  ctx.font = '10px monospace';
  ctx.fillText(`F${String(frameId).padStart(4,'0')} ${showPerson ? '● PERSON' : ''}`, 8, 14);
}

// ─── RSVP Engine ─────────────────────────────────────────────────────────────
function startRSVP() {
  if (state.rsvpInterval) return;
  const intervalMs = 1000 / state.presentationRateHz;

  state.rsvpInterval = setInterval(() => {
    const imgSrc = sampleDroneFrame();
    const ts = performance.timeOrigin + performance.now();
    const frameId = state.frameId++;

    // Display in RSVP window
    document.getElementById('rsvp-img').src = imgSrc;
    document.getElementById('frame-counter').textContent = `F${String(frameId).padStart(4,'0')}`;

    const stimulus = { frameId, ts, imgSrc, score: null, isP300: false };
    state.stimulusLog.push(stimulus);

    // Trim log (keep last 200)
    if (state.stimulusLog.length > 200) state.stimulusLog.shift();

    // Schedule epoch extraction after full post-stimulus window
    setTimeout(() => analyzeFrame(stimulus), EPOCH_POST_MS + 80);

    // In demo mode: if frame has simulated person, tell server to inject P300
    if (state.isDemo) {
      const canvas = document.getElementById('frame-canvas');
      const ctx = canvas.getContext('2d');
      const d = ctx.getImageData(100, 60, 40, 80).data;
      // crude "person detector" — check for warm pixel intensity
      let warmPx = 0;
      for (let i = 0; i < d.length; i += 4) {
        if (d[i] > 150 && d[i+1] > 80 && d[i+2] < 80) warmPx++;
      }
      if (warmPx > 20) {
        wsSend('sim_frame_onset', { ts });
      }
    }
  }, intervalMs);

  log(`RSVP started @ ${state.presentationRateHz}Hz (${Math.round(1000/state.presentationRateHz)}ms/frame)`);
}

function stopRSVP() {
  if (state.rsvpInterval) {
    clearInterval(state.rsvpInterval);
    state.rsvpInterval = null;
    log('RSVP stopped');
  }
}

// ─── P300 Analysis ────────────────────────────────────────────────────────────
function analyzeFrame(stimulus) {
  if (state.mode !== 'scanning') return;

  const epoch = extractEpoch(state.eegBuffer, stimulus.ts);
  const result = scoreEpoch(epoch, state.p300Threshold);

  stimulus.score = result.score;
  stimulus.isP300 = result.isP300;
  stimulus.amplitude = result.amplitude;

  updateSignalBar(result.score);

  if (result.isP300) {
    state.flaggedFrames.unshift({
      ...stimulus,
      detectedAt: Date.now(),
      amplitude: result.amplitude,
    });
    if (state.flaggedFrames.length > 50) state.flaggedFrames.pop();
    renderFlaggedFrames();
    flashDetection();
    log(`P300 detected — frame ${stimulus.frameId} (${result.amplitude}µV, score ${result.score.toFixed(2)})`);
  }
}

// ─── Calibration ─────────────────────────────────────────────────────────────
// Uses an internal image set: 80% non-target, 20% target
const CAL_TARGETS = 15;    // target presentations
const CAL_NONTARGETS = 60; // non-target presentations
const CAL_ISI_MS = 400;    // inter-stimulus interval during calibration

function buildCalibrationSet() {
  const canvas = document.createElement('canvas');
  canvas.width = 320; canvas.height = 240;
  const ctx = canvas.getContext('2d');

  const images = [];

  // Generate non-target frames (empty scenes)
  for (let i = 0; i < CAL_NONTARGETS; i++) {
    drawSimulatedThermalFrame(ctx, 320, 240, i * 5); // no person frames
    images.push({ src: canvas.toDataURL('image/jpeg', 0.7), isTarget: false });
  }

  // Generate target frames (person present)
  for (let i = 0; i < CAL_TARGETS; i++) {
    drawSimulatedThermalFrame(ctx, 320, 240, i * 20); // person frames
    images.push({ src: canvas.toDataURL('image/jpeg', 0.7), isTarget: true });
  }

  // Shuffle (Fisher-Yates)
  for (let i = images.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [images[i], images[j]] = [images[j], images[i]];
  }

  return images;
}

async function runCalibration() {
  state.mode = 'calibrating';
  state.calibrator.reset();
  const images = buildCalibrationSet();

  document.getElementById('cal-progress').style.display = 'block';
  document.getElementById('cal-bar-fill').style.width = '0%';
  log(`Calibration started — ${images.length} stimuli`);

  for (let i = 0; i < images.length; i++) {
    if (state.mode !== 'calibrating') break;

    const img = images[i];
    const ts = performance.timeOrigin + performance.now();

    document.getElementById('rsvp-img').src = img.src;
    document.getElementById('cal-bar-fill').style.width = `${Math.round((i+1)/images.length*100)}%`;
    document.getElementById('cal-label').textContent = `Calibrating ${i+1}/${images.length}`;

    // In demo mode, trigger simulated P300 for targets
    if (state.isDemo && img.isTarget) wsSend('sim_frame_onset', { ts });

    await sleep(CAL_ISI_MS);

    // Extract epoch after ISI
    const epoch = extractEpoch(state.eegBuffer, ts);
    state.calibrator.addEpoch(epoch, img.isTarget);
  }

  if (state.mode === 'calibrating') {
    const result = state.calibrator.compute();
    if (result) {
      state.p300Threshold = result.threshold;
      document.getElementById('threshold-val').textContent = `${result.threshold} µV`;
      document.getElementById('p300-peak-val').textContent = `${result.p300PeakMs} ms`;
      log(`Calibration complete — threshold: ${result.threshold}µV, P300 peak: ${result.p300PeakMs}ms`);
      renderCalibrationResult(result);
    } else {
      log('Calibration failed — insufficient signal. Try again.');
    }
    document.getElementById('cal-progress').style.display = 'none';
    setMode('idle');
  }
}

function renderCalibrationResult(result) {
  const canvas = document.getElementById('cal-plot');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  const timePoints = result.targetAvg.map(s => s.t);
  const tAvg = result.targetAvg.map(s => (s.ch[3] + s.ch[4]) / 2);  // PO3+PO4 avg
  const ntAvg = result.nonTargetAvg.map(s => (s.ch[3] + s.ch[4]) / 2);

  const allVals = [...tAvg, ...ntAvg];
  const minV = Math.min(...allVals);
  const maxV = Math.max(...allVals);
  const range = maxV - minV || 1;

  const mapT = t => (t + 100) / 700 * (W - 40) + 20;
  const mapV = v => H - 20 - ((v - minV) / range) * (H - 40);

  // P300 window shading
  ctx.fillStyle = 'rgba(255,140,0,0.08)';
  ctx.fillRect(mapT(250), 0, mapT(550) - mapT(250), H);

  // Zero line
  ctx.strokeStyle = 'rgba(255,255,255,0.15)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(20, mapV(0));
  ctx.lineTo(W - 20, mapV(0));
  ctx.stroke();

  // Non-target trace
  ctx.strokeStyle = 'rgba(100,160,255,0.7)';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  timePoints.forEach((t, i) => {
    if (i === 0) ctx.moveTo(mapT(t), mapV(ntAvg[i]));
    else ctx.lineTo(mapT(t), mapV(ntAvg[i]));
  });
  ctx.stroke();

  // Target trace (P300)
  ctx.strokeStyle = 'rgba(255,140,0,0.9)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  timePoints.forEach((t, i) => {
    if (i === 0) ctx.moveTo(mapT(t), mapV(tAvg[i]));
    else ctx.lineTo(mapT(t), mapV(tAvg[i]));
  });
  ctx.stroke();

  // Labels
  ctx.fillStyle = 'rgba(255,140,0,0.9)';
  ctx.font = '10px monospace';
  ctx.fillText('TARGET', W - 70, 18);
  ctx.fillStyle = 'rgba(100,160,255,0.9)';
  ctx.fillText('NON-TARGET', W - 90, 32);
}

// ─── EEG Plot ─────────────────────────────────────────────────────────────────
function updateEEGPlot(packet) {
  // Average PO3 + PO4 from each sample in packet for the mini waveform
  for (const sample of packet.data) {
    const avg = (sample[3] + sample[4]) / 2;
    state.eegPlotBuffer.push(avg);
  }
  if (state.eegPlotBuffer.length > 512) {
    state.eegPlotBuffer.splice(0, state.eegPlotBuffer.length - 512);
  }
  drawEEGWaveform();
}

function drawEEGWaveform() {
  const canvas = document.getElementById('eeg-plot');
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const data = state.eegPlotBuffer;

  ctx.clearRect(0, 0, W, H);

  if (data.length < 2) return;

  const minV = -15, maxV = 15;
  const mapV = v => H/2 - (v / (maxV - minV) * H * 0.8);

  // Zero line
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(0, H/2);
  ctx.lineTo(W, H/2);
  ctx.stroke();

  // Waveform
  ctx.strokeStyle = '#ff8c00';
  ctx.lineWidth = 1;
  ctx.beginPath();
  data.forEach((v, i) => {
    const x = (i / data.length) * W;
    const y = mapV(Math.max(minV, Math.min(maxV, v)));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Current value
  const last = data[data.length - 1];
  ctx.fillStyle = last > state.p300Threshold ? '#ff4444' : '#ff8c00';
  ctx.font = '9px monospace';
  ctx.fillText(`${last.toFixed(1)}µV`, W - 45, 12);
}

function updateSignalBar(score) {
  const bar = document.getElementById('p300-bar-fill');
  bar.style.width = `${Math.round(score * 100)}%`;
  bar.style.background = score > 0.5 ? '#ff4444' : score > 0.3 ? '#ff8c00' : '#ff8c00';
  document.getElementById('p300-score').textContent = score.toFixed(2);
}

// ─── UI ───────────────────────────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;
  const modes = ['idle', 'calibrating', 'scanning', 'paused'];
  modes.forEach(m => document.getElementById(`mode-${m}`)?.classList.toggle('active', m === mode));

  document.getElementById('btn-scan').disabled = mode === 'scanning' || mode === 'calibrating';
  document.getElementById('btn-pause').disabled = mode !== 'scanning';
  document.getElementById('btn-cal').disabled = mode === 'scanning' || mode === 'calibrating';
  document.getElementById('mode-label').textContent = mode.toUpperCase();

  if (mode === 'scanning') startRSVP();
  if (mode === 'idle' || mode === 'paused') stopRSVP();
}

function renderFlaggedFrames() {
  const container = document.getElementById('flagged-grid');
  container.innerHTML = '';

  state.flaggedFrames.slice(0, 20).forEach((f, idx) => {
    const el = document.createElement('div');
    el.className = 'flagged-card' + (idx === 0 ? ' new' : '');
    el.innerHTML = `
      <img src="${f.imgSrc}" alt="Frame ${f.frameId}">
      <div class="flagged-info">
        <span class="flagged-id">F${String(f.frameId).padStart(4,'0')}</span>
        <span class="flagged-amp">${f.amplitude}µV</span>
      </div>
    `;
    el.title = `Frame ${f.frameId} — P300 amplitude: ${f.amplitude}µV`;
    container.appendChild(el);

    setTimeout(() => el.classList.remove('new'), 50);
  });

  document.getElementById('detection-count').textContent = state.flaggedFrames.length;
}

function flashDetection() {
  const rsvp = document.getElementById('rsvp-container');
  rsvp.classList.add('flash');
  setTimeout(() => rsvp.classList.remove('flash'), 300);
}

function setStatusDot(id, color) {
  const el = document.getElementById(id);
  if (el) el.setAttribute('data-state', color);
}

function log(msg) {
  const el = document.getElementById('log');
  const ts = new Date().toTimeString().slice(0, 8);
  const line = document.createElement('div');
  line.textContent = `[${ts}] ${msg}`;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
  if (el.children.length > 100) el.removeChild(el.firstChild);
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ─── Event bindings ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  connectWS();

  document.getElementById('btn-cal').addEventListener('click', () => {
    setMode('calibrating');
    runCalibration();
  });

  document.getElementById('btn-scan').addEventListener('click', () => setMode('scanning'));
  document.getElementById('btn-pause').addEventListener('click', () => setMode('paused'));
  document.getElementById('btn-reset').addEventListener('click', () => {
    state.flaggedFrames = [];
    state.frameId = 0;
    renderFlaggedFrames();
    setMode('idle');
    log('Session reset');
  });

  document.getElementById('rate-slider').addEventListener('input', (e) => {
    state.presentationRateHz = parseInt(e.target.value);
    document.getElementById('rate-val').textContent = `${state.presentationRateHz} Hz`;
    if (state.rsvpInterval) {
      stopRSVP();
      startRSVP();
    }
  });

  document.getElementById('threshold-slider').addEventListener('input', (e) => {
    state.p300Threshold = parseFloat(e.target.value);
    document.getElementById('threshold-val').textContent = `${state.p300Threshold.toFixed(1)} µV`;
  });

  // Resize canvases on load
  const resizeCanvases = () => {
    const eegPlot = document.getElementById('eeg-plot');
    const container = eegPlot.parentElement;
    eegPlot.width = container.clientWidth;

    const calPlot = document.getElementById('cal-plot');
    const calContainer = calPlot.parentElement;
    calPlot.width = calContainer.clientWidth;
  };
  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);

  setMode('idle');
  log('NeuroRSVP initialized — calibrate before scanning');
});
