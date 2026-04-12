const PRE_MS = 200;
const POST_MS = 800;
const P300_START_MS = 300;
const P300_END_MS = 600;
const ARTIFACT_UV = 100;

const LABELS = {
  human_fire: { name: 'Humans in fire', short: 'HUMAN', rank: 2 },
  item_fire: { name: 'Items in fire', short: 'ITEM', rank: 1 },
  normal: { name: 'Normal items', short: 'NORMAL', rank: 0 },
};

class CollectionEEGBuffer {
  constructor(maxDurationMs = 12000) {
    this.samples = [];
    this.maxDurationMs = maxDurationMs;
    this.sampleRate = 128;
    this.channels = [];
  }

  push(packet) {
    if (!packet?.data?.length) return;
    const sampleRate = packet.sampleRate || this.sampleRate;
    const dt = 1000 / sampleRate;
    this.sampleRate = sampleRate;
    this.channels = packet.channels || this.channels;

    for (let i = 0; i < packet.data.length; i++) {
      this.samples.push({
        t: packet.startTime + i * dt,
        ch: packet.data[i].map(Number),
      });
    }

    const cutoff = Date.now() - this.maxDurationMs;
    while (this.samples.length && this.samples[0].t < cutoff) this.samples.shift();
  }

  window(startMs, endMs) {
    return this.samples.filter(sample => sample.t >= startMs && sample.t <= endMs);
  }
}

const state = {
  ws: null,
  mode: 'idle',
  isDemo: false,
  buffer: new CollectionEEGBuffer(),
  eegPlot: [],
  categories: [],
  stimuli: [],
  trials: [],
  events: [],
  currentIndex: 0,
  runToken: 0,
  sessionId: makeSessionId(),
  savedPath: null,
};

const $ = (id) => document.getElementById(id);

function makeSessionId() {
  const stamp = new Date().toISOString().replace(/[-:.TZ]/g, '').slice(0, 14);
  return `collection-${stamp}-${Math.random().toString(16).slice(2, 8)}`;
}

function setDot(id, color) {
  $(id)?.setAttribute('data-state', color);
}

function setMode(mode) {
  state.mode = mode;
  $('mode-label').textContent = mode.toUpperCase();
  $('btn-start').disabled = !state.stimuli.length || mode === 'running' || mode === 'paused';
  $('btn-pause').disabled = mode !== 'running' && mode !== 'paused';
  $('btn-pause').textContent = mode === 'paused' ? 'Resume' : 'Pause';
  const hasEvents = state.events.length > 0;
  $('btn-save').disabled = !hasEvents;
  $('btn-json').disabled = !hasEvents;
  $('btn-csv').disabled = !hasEvents;
}

function logRow(message, kind = '') {
  const row = document.createElement('div');
  row.className = 'event-row';
  row.innerHTML = `<b>${kind || 'INFO'}</b><span>${message}</span><span></span>`;
  $('event-log').prepend(row);
  while ($('event-log').children.length > 14) $('event-log').lastChild.remove();
}

function connectWS() {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  state.ws = new WebSocket(`${protocol}://${location.host}`);

  state.ws.onopen = () => {
    $('ws-status').textContent = 'connected';
    setDot('ws-dot', 'green');
  };

  state.ws.onclose = () => {
    $('ws-status').textContent = 'reconnecting';
    setDot('ws-dot', 'red');
    setTimeout(connectWS, 1500);
  };

  state.ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'init') {
      state.isDemo = !!msg.payload.demo;
    }
    if (msg.type === 'crown_status') {
      const status = msg.payload || {};
      $('eeg-status').textContent = status.message || status.state || 'eeg';
      setDot('eeg-dot', status.state === 'online' || status.state === 'demo' ? 'green' : status.state === 'error' ? 'red' : 'amber');
    }
    if (msg.type === 'eeg_packet') {
      state.buffer.push(msg.payload);
      appendWaveform(msg.payload);
    }
  };
}

function wsSend(type, payload = {}) {
  if (state.ws?.readyState === 1) state.ws.send(JSON.stringify({ type, ...payload }));
}

async function loadStimuli() {
  const res = await fetch('/api/collection/stimuli');
  if (!res.ok) throw new Error(`Could not load stimuli: ${res.status}`);
  const data = await res.json();
  state.categories = data.categories || [];
  state.stimuli = state.categories.flatMap(category => category.stimuli);
  $('dataset-count').textContent = `${state.stimuli.length} images`;
  $('session-id').textContent = state.sessionId;
  renderDataset();
  renderStats();
  setMode('idle');
}

function renderDataset() {
  $('dataset-list').innerHTML = state.categories.map(category => `
    <div class="dataset-row">
      <strong>${category.displayName}</strong>
      <span>${category.count}</span>
    </div>
  `).join('');
}

function numericInput(id) {
  return Number($(id).value);
}

function buildTrials(totalTrials) {
  const byLabel = Object.fromEntries(state.categories.map(cat => [cat.label, cat.stimuli]));
  const plan = [
    ['human_fire', 0.20],
    ['item_fire', 0.30],
    ['normal', 0.50],
  ];
  const counts = plan.map(([, ratio]) => Math.max(1, Math.round(totalTrials * ratio)));
  while (counts.reduce((a, b) => a + b, 0) > totalTrials) {
    const idx = counts.indexOf(Math.max(...counts));
    if (counts[idx] > 1) counts[idx]--;
  }
  while (counts.reduce((a, b) => a + b, 0) < totalTrials) counts[2]++;

  const trials = [];
  plan.forEach(([label], planIdx) => {
    const pool = byLabel[label] || [];
    if (!pool.length) return;
    for (let i = 0; i < counts[planIdx]; i++) {
      trials.push({ ...pool[i % pool.length], trialId: `${label}-${i + 1}` });
    }
  });
  shuffle(trials);
  return trials.map((trial, index) => ({ ...trial, trialIndex: index + 1 }));
}

function shuffle(items) {
  for (let i = items.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [items[i], items[j]] = [items[j], items[i]];
  }
}

function showPhase(phase, trial = null) {
  $('fixation').style.display = phase === 'fixation' ? 'block' : 'none';
  $('stimulus-img').style.display = phase === 'image' ? 'block' : 'none';
  $('blank-label').style.display = phase === 'blank' || phase === 'ready' ? 'block' : 'none';
  $('blank-label').textContent = phase === 'ready' ? 'Ready' : '';
  $('phase-label').textContent = phase.toUpperCase();
  if (trial) {
    $('trial-label').textContent = `TRIAL ${String(trial.trialIndex).padStart(3, '0')}/${String(state.trials.length).padStart(3, '0')}`;
  }
}

async function startCollection() {
  if (!state.stimuli.length) return;
  const totalTrials = numericInput('total-trials');
  state.trials = buildTrials(totalTrials);
  state.events = [];
  state.currentIndex = 0;
  state.savedPath = null;
  state.runToken++;
  const token = state.runToken;
  setMode('running');
  renderStats();
  logRow(`started ${state.trials.length} trials`, 'RUN');

  for (let i = 0; i < state.trials.length; i++) {
    if (token !== state.runToken) break;
    while (state.mode === 'paused') await sleep(100);
    if (state.mode !== 'running') break;
    state.currentIndex = i;
    await runTrial(state.trials[i], token);
  }

  if (token === state.runToken && state.mode === 'running') {
    setMode('complete');
    showPhase('ready');
    $('blank-label').textContent = 'Collection complete';
    logRow('collection complete', 'DONE');
    await saveSession();
  }
}

async function runTrial(trial, token) {
  const fixationMs = numericInput('fixation-ms');
  const imageMs = numericInput('image-ms');
  const blankMs = numericInput('blank-ms');

  showPhase('fixation', trial);
  updateProgress();
  await sleep(fixationMs);
  if (token !== state.runToken || state.mode !== 'running') return;

  const img = $('stimulus-img');
  img.src = trial.src;
  try { await img.decode(); } catch {}

  const stimulusTs = performance.timeOrigin + performance.now();
  showPhase('image', trial);
  $('current-meta').textContent = `${LABELS[trial.label]?.name || trial.label} | ${trial.fileName}`;

  if (state.isDemo) {
    wsSend('sim_frame_onset', {
      ts: stimulusTs,
      isTarget: trial.isTarget,
      isFace: trial.label === 'human_fire',
      salience: trial.demoSalience,
      label: trial.label,
    });
  }

  await sleep(imageMs);
  if (token !== state.runToken || state.mode !== 'running') return;
  showPhase('blank', trial);
  await sleep(blankMs);

  const waitTarget = stimulusTs + POST_MS + 80;
  while (Date.now() < waitTarget) await sleep(20);

  const event = analyzeTrial(trial, stimulusTs);
  state.events.push(event);
  updateProgress();
  renderStats();
  renderEvent(event);
  autosave();
}

function analyzeTrial(trial, stimulusTs) {
  const epoch = extractEpoch(stimulusTs);
  const threshold = numericInput('threshold-uv');
  const result = scoreEpoch(epoch, threshold);
  const predictedClass = classifyResponse(result, threshold);
  return {
    sessionId: state.sessionId,
    trialIndex: trial.trialIndex,
    stimulusTimestamp: Math.round(stimulusTs),
    collectedAt: new Date().toISOString(),
    imageId: trial.imageId,
    fileName: trial.fileName,
    folder: trial.folder,
    label: trial.label,
    predictedClass,
    isTarget: trial.isTarget,
    p300Detected: result.p300Detected,
    p300Score: result.score,
    confidence: result.confidence,
    p300AmplitudeUv: result.amplitude,
    meanAmplitudeUv: result.meanAmplitude,
    p300LatencyMs: result.latencyMs,
    auc: result.auc,
    thresholdUv: threshold,
    samples: result.samples,
    sampleRate: state.buffer.sampleRate,
    channelsUsed: result.channelsUsed,
    rejected: result.rejected,
    rejectReason: result.rejectReason,
  };
}

function extractEpoch(stimulusTs) {
  const samples = state.buffer.window(stimulusTs - PRE_MS, stimulusTs + POST_MS);
  const expected = Math.round((PRE_MS + POST_MS) / 1000 * state.buffer.sampleRate);
  if (samples.length < expected * 0.60) {
    return { rejected: true, rejectReason: `too few samples (${samples.length}/${expected})`, samples: samples.length };
  }

  const nChannels = Math.max(...samples.map(sample => sample.ch.length));
  const baseline = samples.filter(sample => sample.t < stimulusTs);
  if (baseline.length < 5) return { rejected: true, rejectReason: 'missing baseline', samples: samples.length };

  const means = Array.from({ length: nChannels }, (_, channel) => {
    const vals = baseline.map(sample => sample.ch[channel]).filter(Number.isFinite);
    return vals.length ? vals.reduce((sum, value) => sum + value, 0) / vals.length : 0;
  });

  const epoch = samples.map(sample => ({
    t: sample.t - stimulusTs,
    ch: sample.ch.map((value, channel) => value - means[channel]),
  }));

  for (const sample of epoch) {
    for (const value of sample.ch) {
      if (Math.abs(value) > ARTIFACT_UV) {
        return { rejected: true, rejectReason: `artifact > ${ARTIFACT_UV}uV`, samples: samples.length };
      }
    }
  }

  return { rejected: false, epoch, samples: samples.length };
}

function p300Channels(nChannels) {
  const names = state.buffer.channels || [];
  const pz = names.findIndex(name => String(name).toLowerCase() === 'pz');
  if (pz >= 0) return [pz];
  if (nChannels === 5) return [2];
  if (nChannels >= 8) return [3, 4, 0, 7].filter(channel => channel < nChannels);
  return [Math.max(0, Math.floor(nChannels / 2))];
}

function scoreEpoch(extraction, threshold) {
  if (extraction.rejected) {
    return {
      rejected: true,
      rejectReason: extraction.rejectReason,
      samples: extraction.samples || 0,
      p300Detected: false,
      score: 0,
      confidence: 0,
      amplitude: 0,
      meanAmplitude: 0,
      latencyMs: null,
      auc: 0,
      channelsUsed: [],
    };
  }

  const epoch = extraction.epoch;
  const nChannels = epoch[0]?.ch.length || 0;
  const channels = p300Channels(nChannels);
  const window = epoch.filter(sample => sample.t >= P300_START_MS && sample.t <= P300_END_MS);
  if (!window.length || !channels.length) {
    return { rejected: true, rejectReason: 'empty P300 window', samples: extraction.samples, p300Detected: false, score: 0, confidence: 0, amplitude: 0, meanAmplitude: 0, latencyMs: null, auc: 0, channelsUsed: channels };
  }

  const series = window.map(sample => ({
    t: sample.t,
    value: channels.reduce((sum, channel) => sum + (sample.ch[channel] || 0), 0) / channels.length,
  }));
  const values = series.map(point => point.value);
  let peakIdx = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[peakIdx]) peakIdx = i;
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  let auc = 0;
  for (let i = 1; i < series.length; i++) {
    auc += ((series[i - 1].value + series[i].value) / 2) * ((series[i].t - series[i - 1].t) / 1000);
  }
  const amplitude = values[peakIdx];
  const score = Math.max(0, Math.min(1, amplitude / (threshold * 2)));
  return {
    rejected: false,
    rejectReason: '',
    samples: extraction.samples,
    p300Detected: amplitude >= threshold,
    score: round(score, 3),
    confidence: round(Math.max(score, 1 - score), 3),
    amplitude: round(amplitude, 2),
    meanAmplitude: round(mean, 2),
    latencyMs: Math.round(series[peakIdx].t),
    auc: round(auc, 4),
    channelsUsed: channels.map(channel => state.buffer.channels[channel] || `ch${channel}`),
  };
}

function classifyResponse(result, threshold) {
  if (result.rejected) return 'rejected';
  if (result.amplitude >= threshold * 1.45) return 'human_fire';
  if (result.amplitude >= threshold * 0.75) return 'item_fire';
  return 'normal';
}

function renderEvent(event) {
  const row = document.createElement('div');
  row.className = 'event-row';
  const hitClass = event.p300Detected ? 'hit' : 'miss';
  row.innerHTML = `
    <b>${LABELS[event.label]?.short || event.label}</b>
    <span>${event.imageId}</span>
    <span class="${hitClass}">${event.rejected ? 'reject' : `${event.p300AmplitudeUv}uV`}</span>
  `;
  $('event-log').prepend(row);
  while ($('event-log').children.length > 14) $('event-log').lastChild.remove();
}

function renderStats() {
  const labels = ['human_fire', 'item_fire', 'normal'];
  const rows = labels.map(label => {
    const events = state.events.filter(event => event.label === label && !event.rejected);
    const hits = events.filter(event => event.p300Detected).length;
    const meanAmp = mean(events.map(event => event.p300AmplitudeUv));
    const meanScore = mean(events.map(event => event.p300Score));
    return `
      <div class="stat-row">
        <strong>${LABELS[label].name}</strong>
        <span>${events.length} / hit ${hits} / ${round(meanAmp, 2)}uV / ${round(meanScore, 2)}</span>
      </div>
    `;
  });
  const rejected = state.events.filter(event => event.rejected).length;
  rows.push(`
    <div class="stat-row">
      <strong>Total</strong>
      <span>${state.events.length} rows / ${rejected} rejected</span>
    </div>
  `);
  $('stats-grid').innerHTML = rows.join('');
}

function updateProgress() {
  const total = state.trials.length || numericInput('total-trials') || 1;
  const done = state.events.length;
  $('progress-fill').style.width = `${Math.min(100, Math.round((done / total) * 100))}%`;
  $('progress-text').textContent = `${done}/${total} trials collected`;
}

function appendWaveform(packet) {
  const pzIndex = (packet.channels || []).findIndex(name => String(name).toLowerCase() === 'pz');
  const channel = pzIndex >= 0 ? pzIndex : Math.min(2, (packet.data[0]?.length || 1) - 1);
  for (const sample of packet.data) state.eegPlot.push(sample[channel] || 0);
  while (state.eegPlot.length > 360) state.eegPlot.shift();
  drawWaveform();
}

function drawWaveform() {
  const canvas = $('waveform');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(255,255,255,.12)';
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();
  if (state.eegPlot.length < 2) return;
  ctx.strokeStyle = '#ff842b';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  state.eegPlot.forEach((value, idx) => {
    const x = (idx / (state.eegPlot.length - 1)) * w;
    const y = h / 2 - Math.max(-20, Math.min(20, value)) * 1.8;
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function sessionPayload() {
  return {
    sessionId: state.sessionId,
    createdAt: new Date().toISOString(),
    protocol: {
      totalTrials: state.trials.length,
      fixationMs: numericInput('fixation-ms'),
      imageMs: numericInput('image-ms'),
      blankMs: numericInput('blank-ms'),
      preMs: PRE_MS,
      postMs: POST_MS,
      p300WindowMs: [P300_START_MS, P300_END_MS],
      thresholdUv: numericInput('threshold-uv'),
    },
    dataset: state.categories.map(category => ({
      folder: category.folder,
      label: category.label,
      count: category.count,
    })),
    stats: computeStats(),
    events: state.events,
  };
}

function computeStats() {
  const stats = {};
  for (const label of Object.keys(LABELS)) {
    const events = state.events.filter(event => event.label === label && !event.rejected);
    stats[label] = {
      events: events.length,
      p300Detected: events.filter(event => event.p300Detected).length,
      meanAmplitudeUv: round(mean(events.map(event => event.p300AmplitudeUv)), 3),
      meanScore: round(mean(events.map(event => event.p300Score)), 3),
      meanLatencyMs: round(mean(events.map(event => event.p300LatencyMs).filter(Number.isFinite)), 1),
    };
  }
  stats.rejected = state.events.filter(event => event.rejected).length;
  return stats;
}

async function saveSession() {
  if (!state.events.length) return;
  $('btn-save').disabled = true;
  try {
    const res = await fetch('/api/collection/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sessionPayload()),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `save failed ${res.status}`);
    state.savedPath = data.path;
    logRow(`saved ${data.events} rows`, 'SAVE');
  } catch (err) {
    logRow(err.message, 'ERR');
  } finally {
    setMode(state.mode);
  }
}

function downloadJson() {
  downloadFile(`${state.sessionId}.json`, 'application/json', JSON.stringify(sessionPayload(), null, 2));
}

function downloadCsv() {
  const cols = [
    'sessionId', 'trialIndex', 'stimulusTimestamp', 'collectedAt', 'imageId', 'fileName',
    'folder', 'label', 'predictedClass', 'isTarget', 'p300Detected', 'p300Score',
    'confidence', 'p300AmplitudeUv', 'meanAmplitudeUv', 'p300LatencyMs', 'auc',
    'thresholdUv', 'samples', 'sampleRate', 'channelsUsed', 'rejected', 'rejectReason',
  ];
  const lines = [cols.join(',')];
  for (const event of state.events) {
    lines.push(cols.map(col => csvCell(Array.isArray(event[col]) ? event[col].join('|') : event[col])).join(','));
  }
  downloadFile(`${state.sessionId}.csv`, 'text/csv', lines.join('\n'));
}

function downloadFile(name, type, content) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function csvCell(value) {
  const text = value == null ? '' : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function autosave() {
  try {
    localStorage.setItem('neurorsvp.collection.autosave', JSON.stringify(sessionPayload()));
  } catch {}
}

function resetSession() {
  state.runToken++;
  state.trials = [];
  state.events = [];
  state.currentIndex = 0;
  state.sessionId = makeSessionId();
  state.savedPath = null;
  $('session-id').textContent = state.sessionId;
  $('stimulus-img').removeAttribute('src');
  showPhase('ready');
  $('blank-label').textContent = 'Ready';
  $('current-meta').textContent = 'Each row will align image_id, label, stimulus timestamp, EEG response, and P300 features.';
  updateProgress();
  renderStats();
  $('event-log').innerHTML = '';
  setMode('idle');
}

function mean(values) {
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return 0;
  return finite.reduce((sum, value) => sum + value, 0) / finite.length;
}

function round(value, digits = 2) {
  if (!Number.isFinite(value)) return 0;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

document.addEventListener('DOMContentLoaded', async () => {
  $('session-id').textContent = state.sessionId;
  connectWS();
  try {
    await loadStimuli();
  } catch (err) {
    logRow(err.message, 'ERR');
  }

  $('btn-start').addEventListener('click', () => {
    if (state.mode === 'complete') resetSession();
    startCollection();
  });
  $('btn-pause').addEventListener('click', () => {
    if (state.mode === 'running') setMode('paused');
    else if (state.mode === 'paused') setMode('running');
  });
  $('btn-reset').addEventListener('click', resetSession);
  $('btn-save').addEventListener('click', saveSession);
  $('btn-json').addEventListener('click', downloadJson);
  $('btn-csv').addEventListener('click', downloadCsv);

  setMode('idle');
  showPhase('ready');
});
