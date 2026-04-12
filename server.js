import { config } from 'dotenv';
config();
import express from 'express';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import fs from 'fs';
import { promises as fsp } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { CortexClient } from './cortex.js';
import { SyntheticEEGStream, generateTrainingDataset, datasetStats } from './eeg_synthetic.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app    = express();
const server = createServer(app);
const wss    = new WebSocketServer({ server });

app.use(express.static(path.join(__dirname)));
app.use(express.json({ limit: '10mb' }));

app.get('/review', (_req, res) => res.sendFile(path.join(__dirname, 'review.html')));
app.get('/collection', (_req, res) => res.sendFile(path.join(__dirname, 'collection.html')));

function detectImageContentType(filePath) {
  try {
    const header = fs.readFileSync(filePath).subarray(0, 16);
    if (header[0] === 0xff && header[1] === 0xd8) return 'image/jpeg';
    if (header[0] === 0x89 && header[1] === 0x50 && header[2] === 0x4e && header[3] === 0x47) return 'image/png';
    if (header.toString('ascii', 0, 4) === 'RIFF' && header.toString('ascii', 8, 12) === 'WEBP') return 'image/webp';
  } catch {}
  return 'application/octet-stream';
}

function collectionImagePath(category, fileName) {
  if (!COLLECTION_CATEGORIES[category]) return null;
  const categoryDir = path.resolve(__dirname, 'images', category);
  const filePath = path.resolve(categoryDir, fileName);
  if (!filePath.startsWith(categoryDir + path.sep)) return null;
  if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return null;
  return filePath;
}

function listCollectionStimuli() {
  return Object.entries(COLLECTION_CATEGORIES).map(([folder, cfg]) => {
    const dir = path.join(__dirname, 'images', folder);
    const files = fs.existsSync(dir)
      ? fs.readdirSync(dir, { withFileTypes: true })
          .filter(entry => entry.isFile())
          .map(entry => entry.name)
          .sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }))
      : [];
    const stimuli = files.map((fileName, idx) => ({
      id: `${cfg.label}-${idx + 1}`,
      fileName,
      imageId: path.parse(fileName).name || fileName,
      folder,
      label: cfg.label,
      displayName: cfg.displayName,
      oddballWeight: cfg.oddballWeight,
      demoSalience: cfg.demoSalience,
      isTarget: cfg.isTarget,
      src: `/api/collection/image/${encodeURIComponent(folder)}/${encodeURIComponent(fileName)}`,
    }));
    return { folder, ...cfg, count: stimuli.length, stimuli };
  });
}

// ── Session store ─────────────────────────────────────────────────────────────
const sessionDetections = [];
const telemetryLog      = [];
let   latestTelemetry   = null;
const collectionSessions = new Map();

let connectedClients = new Set();
let lastCrownStatus  = null;
let cortex           = null;
let latestMusePacketAt = null;
let latestMuseBands = null;
let latestMuseStatusBroadcastAt = 0;

const COLLECTION_CATEGORIES = {
  'humans in fire': {
    label: 'human_fire',
    displayName: 'Humans in fire',
    oddballWeight: 0.20,
    demoSalience: 1.7,
    isTarget: true,
  },
  'items in fire': {
    label: 'item_fire',
    displayName: 'Items in fire',
    oddballWeight: 0.30,
    demoSalience: 1.0,
    isTarget: true,
  },
  'abnormal items': {
    label: 'normal',
    displayName: 'Normal items',
    oddballWeight: 0.50,
    demoSalience: 0.0,
    isTarget: false,
  },
};

const CV_PROMPT = `Firefighting SAR drone frame. Respond ONLY with valid JSON:
{"personDetected":boolean,"confidence":number,"objects":[],"framePosition":"top-left|top-center|top-right|center-left|center|center-right|bottom-left|bottom-center|bottom-right","estimatedCount":number,"thermalSignature":"string","notes":"string"}`;

const TRIP_SUMMARY_PROMPT = `You are writing an after-action summary for firefighters reviewing a drone + EEG scan.
Use only the supplied JSON. Do not invent victims, locations, or hazards.
Write concise operational prose with these sections:
1. Situation summary
2. People / victim candidates
3. Fire and smoke events
4. EEG corroboration
5. Recommended next actions
Mention uncertainty when detections are weak or GPS is missing.`;

function getCvProvider() {
  const requested = (process.env.CV_ANALYSIS_PROVIDER || 'auto').toLowerCase();
  if (requested === 'openai' && process.env.OPENAI_API_KEY) return 'openai';
  if (requested === 'anthropic' && process.env.ANTHROPIC_API_KEY) return 'anthropic';
  if (process.env.OPENAI_API_KEY) return 'openai';
  if (process.env.ANTHROPIC_API_KEY) return 'anthropic';
  return null;
}

function getEegSource() {
  return (process.env.EEG_SOURCE || 'muse').toLowerCase();
}

function getConfigPayload() {
  const cvProvider = getCvProvider();
  const eegSource = getEegSource();
  return {
    droneUrl:         process.env.DRONE_STREAM_URL || null,
    demo:             eegSource === 'demo' || (eegSource === 'emotiv' && (!process.env.EMOTIV_CLIENT_ID || process.env.EMOTIV_CLIENT_ID === 'your-client-id')),
    eegSource,
    hasAnthropicKey:  !!process.env.ANTHROPIC_API_KEY,
    hasOpenAIKey:     !!process.env.OPENAI_API_KEY,
    hasAiAnalysisKey: !!cvProvider,
    cvProvider:       cvProvider || 'demo',
    headset:          eegSource === 'emotiv' ? 'Emotiv Insight' : eegSource === 'demo' ? 'Synthetic EEG' : 'Muse 2 via LSL bridge',
  };
}

function parseCvJson(text) {
  try {
    return JSON.parse(text.replace(/```json|```/g, '').trim());
  } catch {
    return { personDetected: false, confidence: 0, notes: text, parseError: true };
  }
}

async function analyzeWithAnthropic(base64) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type':      'application/json',
      'x-api-key':         process.env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model:      process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-20250514',
      max_tokens: 400,
      messages: [{
        role: 'user',
        content: [
          { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: base64 } },
          { type: 'text',  text: CV_PROMPT },
        ],
      }],
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || `Anthropic error ${response.status}`);
  return parseCvJson(data.content?.map(b => b.text || '').join('') || '{}');
}

async function analyzeWithOpenAI(imageDataUrl) {
  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: process.env.OPENAI_MODEL || 'gpt-4.1-mini',
      input: [{
        role: 'user',
        content: [
          { type: 'input_text', text: CV_PROMPT },
          { type: 'input_image', image_url: imageDataUrl },
        ],
      }],
      max_output_tokens: 400,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || `OpenAI error ${response.status}`);
  const text = data.output_text || data.output?.flatMap(item => item.content || []).map(part => part.text || '').join('') || '{}';
  return parseCvJson(text);
}

async function summarizeWithOpenAI(report) {
  const response = await fetch('https://api.openai.com/v1/responses', {
    method: 'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({
      model: process.env.OPENAI_MODEL || 'gpt-4.1-mini',
      input: [{
        role: 'user',
        content: [{
          type: 'input_text',
          text: `${TRIP_SUMMARY_PROMPT}\n\nTrip JSON:\n${JSON.stringify(report, null, 2)}`,
        }],
      }],
      max_output_tokens: 900,
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || `OpenAI error ${response.status}`);
  return data.output_text || data.output?.flatMap(item => item.content || []).map(part => part.text || '').join('') || '';
}

async function summarizeWithAnthropic(report) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type':      'application/json',
      'x-api-key':         process.env.ANTHROPIC_API_KEY,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: process.env.ANTHROPIC_MODEL || 'claude-sonnet-4-20250514',
      max_tokens: 900,
      messages: [{
        role: 'user',
        content: `${TRIP_SUMMARY_PROMPT}\n\nTrip JSON:\n${JSON.stringify(report, null, 2)}`,
      }],
    }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error?.message || `Anthropic error ${response.status}`);
  return data.content?.map(b => b.text || '').join('') || '';
}

function formatTripSummaryFallback(report) {
  const s = report.summary || {};
  const lines = s.lines?.length ? s.lines : ['No trip events have been recorded yet.'];
  const risks = report.risks || [];
  const victims = report.victims || [];
  const eegHits = (report.timeline || []).filter(e => e.type === 'eeg_p300');

  return [
    '1. Situation summary',
    ...lines.map(line => `- ${line}`),
    '',
    '2. People / victim candidates',
    victims.length
      ? victims.map(v => `- ${v.victim_id}: priority ${Math.round(v.priority * 100)}%, confidence ${Math.round(v.max_confidence * 100)}%, ${v.detections} detection(s), GPS ${v.gps?.lat != null ? `${v.gps.lat.toFixed(5)}, ${v.gps.lon.toFixed(5)}` : 'unknown'}.`).join('\n')
      : '- No persistent victim candidates were detected.',
    '',
    '3. Fire and smoke events',
    risks.length
      ? risks.map(r => `- ${r.kind.toUpperCase()}: severity ${(r.max_severity * 100).toFixed(1)}%, ${r.frame_count} frame(s), duration ${r.duration_s}s, GPS ${r.gps?.lat != null ? `${r.gps.lat.toFixed(5)}, ${r.gps.lon.toFixed(5)}` : 'unknown'}.`).join('\n')
      : '- No fire or smoke events were detected.',
    '',
    '4. EEG corroboration',
    eegHits.length
      ? `- ${eegHits.length} firefighter P300 hit(s) were logged during the trip.`
      : '- No firefighter P300 hits were logged.',
    '',
    '5. Recommended next actions',
    risks.length || victims.length
      ? '- Re-fly the highest-risk areas slowly, prioritize confirmed person candidates, and verify any GPS-unknown hazards manually.'
      : '- Continue scanning with a stable mirrored feed and broad coverage until the detector has enough frames to summarize.',
  ].join('\n');
}

// ── Broadcast helpers ─────────────────────────────────────────────────────────
function broadcast(type, payload) {
  const msg = JSON.stringify({ type, payload, ts: Date.now() });
  for (const client of connectedClients) {
    if (client.readyState === 1) client.send(msg);
  }
}

function broadcastStatus(payload) {
  lastCrownStatus = payload;
  broadcast('crown_status', payload);
}

// ── EEG source connection ─────────────────────────────────────────────────────
async function connectEEGSource() {
  const eegSource = getEegSource();
  if (eegSource === 'muse') {
    console.log('[Muse] Waiting for Muse 2 LSL bridge at /api/muse/eeg');
    broadcastStatus({ state: 'connecting', message: 'Waiting for Muse 2 LSL bridge — run: python muse_lsl_bridge.py' });
    return;
  }
  if (eegSource === 'demo') {
    console.log('[EEG] Running in DEMO mode');
    broadcastStatus({ state: 'demo', message: 'Demo mode — simulated EEG active' });
    startSimulatedEEG();
    return;
  }
  await connectInsight();
}

// ── Emotiv Cortex connection ──────────────────────────────────────────────────
async function connectInsight() {
  const clientId     = process.env.EMOTIV_CLIENT_ID;
  const clientSecret = process.env.EMOTIV_CLIENT_SECRET;

  if (!clientId || clientId === 'your-client-id') {
    console.log('[Insight] No credentials — running in DEMO mode');
    broadcastStatus({ state: 'demo', message: 'Demo mode — simulated EEG active' });
    startSimulatedEEG();
    return;
  }

  cortex = new CortexClient({
    clientId,
    clientSecret,
    onStatus: (msg) => {
      console.log('[Cortex]', msg);
      broadcastStatus({ state: 'connecting', message: msg });
    },
    onEEGPacket: (packet) => {
      broadcast('eeg_packet', packet);
    },
  });

  try {
    await cortex.connect();
    broadcastStatus({ state: 'online', message: 'Emotiv Insight streaming — 128Hz · AF3 T7 Pz T8 AF4' });
  } catch (err) {
    console.error('[Cortex] Failed:', err.message);
    broadcastStatus({ state: 'error', message: err.message });
    console.log('[Cortex] Falling back to demo mode');
    startSimulatedEEG();
  }
}

// ── Simulated EEG (demo mode) ─────────────────────────────────────────────────
let syntheticStream = null;

function startSimulatedEEG() {
  syntheticStream = new SyntheticEEGStream({
    controllerActivity: 0.5,
    onStatus: (msg) => broadcastStatus({ state: 'demo', message: msg }),
    onPacket: (packet) => broadcast('eeg_packet', packet),
  });
  syntheticStream.start();
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
wss.on('connection', (ws) => {
  connectedClients.add(ws);
  console.log(`[WS] Client connected (${connectedClients.size} total)`);

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw);
      if (msg.type === 'sim_frame_onset' && syntheticStream) {
        const isTarget = msg.isTarget ?? msg.containsPerson ?? msg.isFace ?? false;
        const salience = Number.isFinite(msg.salience) ? msg.salience : (isTarget ? 1 : 0);
        syntheticStream.notifyFrameOnset(msg.ts, isTarget, salience);
      }
      if (msg.type === 'ping') ws.send(JSON.stringify({ type: 'pong', ts: Date.now() }));
    } catch {}
  });

  ws.on('close', () => {
    connectedClients.delete(ws);
    console.log(`[WS] Client disconnected (${connectedClients.size} total)`);
  });

  ws.send(JSON.stringify({
    type: 'init',
    payload: getConfigPayload(),
  }));

  // Replay last known status so browser doesn't show stale white dot
  if (lastCrownStatus) {
    ws.send(JSON.stringify({ type: 'crown_status', payload: lastCrownStatus, ts: Date.now() }));
  }
});

// ── REST: Config ──────────────────────────────────────────────────────────────
app.get('/api/config', (req, res) => {
  res.json(getConfigPayload());
});

// ── REST: Muse 2 LSL bridge ──────────────────────────────────────────────────
app.post('/api/muse/eeg', (req, res) => {
  const packet = req.body || {};
  if (!Array.isArray(packet.data) || !packet.data.length) {
    return res.status(400).json({ error: 'Expected data[] EEG samples' });
  }
  const sampleRate = Number(packet.sampleRate) > 0 ? Number(packet.sampleRate) : 256;
  const startTime = Number(packet.startTime) || Date.now();
  const channels = Array.isArray(packet.channels) && packet.channels.length
    ? packet.channels.map(String)
    : packet.data[0].map((_v, idx) => `Ch${idx + 1}`);

  const payload = {
    data: packet.data,
    startTime,
    sampleRate,
    channels,
    source: 'muse2-lsl',
  };
  latestMusePacketAt = Date.now();
  broadcast('eeg_packet', payload);
  if (Date.now() - latestMuseStatusBroadcastAt > 1000) {
    broadcastStatus({ state: 'online', message: `Muse 2 streaming — ${sampleRate.toFixed(0)}Hz · ${channels.join(' ')}` });
    latestMuseStatusBroadcastAt = Date.now();
  }
  res.json({ ok: true, samples: payload.data.length, channels, sampleRate });
});

app.post('/api/muse/bands', (req, res) => {
  latestMuseBands = {
    receivedAt: Date.now(),
    ...req.body,
  };
  broadcast('muse_bands', latestMuseBands);
  res.json({ ok: true });
});

app.get('/api/muse/status', (_req, res) => {
  res.json({
    source: getEegSource(),
    online: latestMusePacketAt ? Date.now() - latestMusePacketAt < 5000 : false,
    lastPacketAt: latestMusePacketAt,
    bands: latestMuseBands,
  });
});

// ── REST: EEG collection image trials ────────────────────────────────────────
app.get('/api/collection/stimuli', (_req, res) => {
  const categories = listCollectionStimuli();
  res.json({
    categories,
    total: categories.reduce((sum, cat) => sum + cat.count, 0),
    ratios: {
      human_fire: COLLECTION_CATEGORIES['humans in fire'].oddballWeight,
      item_fire: COLLECTION_CATEGORIES['items in fire'].oddballWeight,
      normal: COLLECTION_CATEGORIES['abnormal items'].oddballWeight,
    },
  });
});

app.get('/api/collection/image/:category/:fileName', (req, res) => {
  const filePath = collectionImagePath(req.params.category, req.params.fileName);
  if (!filePath) return res.status(404).send('Image not found');
  res.type(detectImageContentType(filePath));
  res.sendFile(filePath);
});

app.post('/api/collection/sessions', async (req, res) => {
  const session = req.body || {};
  if (!session.sessionId || !Array.isArray(session.events)) {
    return res.status(400).json({ error: 'Expected sessionId and events[]' });
  }

  const saved = {
    ...session,
    receivedAt: new Date().toISOString(),
  };
  collectionSessions.set(session.sessionId, saved);

  const outDir = path.join(__dirname, 'collection_data');
  await fsp.mkdir(outDir, { recursive: true });
  const safeId = String(session.sessionId).replace(/[^a-zA-Z0-9_-]/g, '-').slice(0, 80);
  const outPath = path.join(outDir, `${safeId}.json`);
  await fsp.writeFile(outPath, JSON.stringify(saved, null, 2), 'utf8');

  res.json({ ok: true, sessionId: session.sessionId, events: session.events.length, path: outPath });
});

app.get('/api/collection/sessions/:sessionId', (req, res) => {
  const session = collectionSessions.get(req.params.sessionId);
  if (!session) return res.status(404).json({ error: 'Collection session not found in memory' });
  res.json(session);
});

// ── REST: Training data ───────────────────────────────────────────────────────
// GET /api/training-data?n=200&controllerActivity=0.5
// Returns a labeled dataset of synthetic N170/VPP epochs for classifier training.
app.get('/api/training-data', (req, res) => {
  const n                  = Math.min(parseInt(req.query.n || '200', 10), 1000);
  const controllerActivity = parseFloat(req.query.controllerActivity || '0.5');
  const keepArtifacted     = req.query.keepArtifacted === 'true';

  const trials = generateTrainingDataset(n, { controllerActivity, keepArtifacted });
  const stats  = datasetStats(trials);

  res.json({
    stats,
    trials: trials.map(t => ({
      label:        t.label,
      isFace:       t.isFace,
      isArtifacted: t.isArtifacted,
      features:     t.features,
      epochMatrix:  t.epoch.map(s => Array.from(s.ch)),  // [nSamples × 5]
      epochTimes:   t.epoch.map(s => s.t),               // ms relative to onset
    })),
  });
});

// ── REST: Detections ──────────────────────────────────────────────────────────
app.post('/api/detections', (req, res) => {
  const { frameId, ts, imgSrc, amplitude, score, gps } = req.body;
  if (!sessionDetections.find(d => d.frameId === frameId)) {
    sessionDetections.push({
      frameId, ts, imgSrc, amplitude, score,
      gps:      gps || latestTelemetry || null,
      cv:       null,
      reviewed: false,
      confirmed: false,
    });
    console.log(`[Session] Detection logged: frame ${frameId} (${amplitude}µV)`);
  }
  res.json({ ok: true, total: sessionDetections.length });
});

app.get('/api/detections',       (req, res) => res.json(sessionDetections));
app.delete('/api/detections',    (req, res) => { sessionDetections.length = 0; res.json({ ok: true }); });

app.patch('/api/detections/:frameId', (req, res) => {
  const det = sessionDetections.find(d => d.frameId === parseInt(req.params.frameId));
  if (!det) return res.status(404).json({ error: 'Not found' });
  Object.assign(det, req.body);
  broadcast('detection_updated', det);
  res.json(det);
});

// ── REST: Telemetry ───────────────────────────────────────────────────────────
app.post('/api/telemetry', (req, res) => {
  const entry = { ts: Date.now(), ...req.body };
  latestTelemetry = entry;
  telemetryLog.push(entry);
  if (telemetryLog.length > 10000) telemetryLog.shift();
  broadcast('telemetry', entry);
  res.json({ ok: true });
});

app.get('/api/telemetry', (req, res) => res.json(telemetryLog));

// ── REST: CV Analysis ─────────────────────────────────────────────────────────
app.post('/api/analyze/:frameId', async (req, res) => {
  const frameId = parseInt(req.params.frameId);
  const det = sessionDetections.find(d => d.frameId === frameId);
  if (!det) return res.status(404).json({ error: 'Detection not found' });

  const provider = getCvProvider();
  if (!provider) {
    const demo = {
      personDetected: true, confidence: 0.87,
      objects: ['person', 'possible victim'],
      framePosition: 'center-left', estimatedCount: 1,
      thermalSignature: 'high — consistent with living person',
      notes: 'Demo mode — set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env for real analysis',
      demo: true,
    };
    det.cv = demo;
    return res.json(demo);
  }

  try {
    const base64 = det.imgSrc.replace(/^data:image\/\w+;base64,/, '');
    const cv = provider === 'openai'
      ? await analyzeWithOpenAI(det.imgSrc)
      : await analyzeWithAnthropic(base64);
    cv.provider = provider;
    det.cv = cv;
    console.log(`[CV] Frame ${frameId}: provider=${provider}, person=${cv.personDetected}, confidence=${cv.confidence}`);
    res.json(cv);
  } catch (err) {
    console.error('[CV] Failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── REST: LLM Trip Summary ───────────────────────────────────────────────────
app.post('/api/trip-summary', async (req, res) => {
  const provider = getCvProvider();
  let report = req.body?.report || null;

  if (!report) {
    try {
      const detectorUrl = process.env.DETECTOR_URL || 'http://127.0.0.1:8000';
      const detectorRes = await fetch(`${detectorUrl}/insights`);
      if (detectorRes.ok) report = await detectorRes.json();
    } catch {}
  }

  if (!report) return res.status(400).json({ error: 'No trip report available yet' });

  const compactReport = {
    generatedAt: new Date().toISOString(),
    summary: report.summary || {},
    victims: (report.victims || []).slice(0, 20),
    risks: (report.risks || []).slice(0, 30),
    timeline: (report.timeline || []).slice(-60),
    heatmap: (report.heatmap || []).slice(0, 30),
    importantFrames: (report.important_frames || []).slice(0, 24).map(({ thumbnail, ...frame }) => frame),
    p300Detections: sessionDetections.map(d => ({
      frameId: d.frameId,
      ts: d.ts,
      amplitude: d.amplitude,
      score: d.score,
      gps: d.gps || null,
      cv: d.cv || null,
      confirmed: d.confirmed,
    })).slice(-50),
    telemetrySamples: telemetryLog.slice(-20),
  };

  if (!provider) {
    return res.json({
      ok: true,
      provider: 'demo',
      summary: formatTripSummaryFallback(compactReport),
      report: compactReport,
    });
  }

  try {
    const summary = provider === 'openai'
      ? await summarizeWithOpenAI(compactReport)
      : await summarizeWithAnthropic(compactReport);
    res.json({ ok: true, provider, summary, report: compactReport });
  } catch (err) {
    console.error('[TripSummary] Failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
server.listen(PORT, async () => {
  console.log(`\n╔══════════════════════════════════════╗`);
  console.log(`║   NeuroRSVP — EEG Image Collection  ║`);
  console.log(`║   http://localhost:${PORT}               ║`);
  console.log(`║   Review: http://localhost:${PORT}/review║`);
  console.log(`║   Collect: http://localhost:${PORT}/collection║`);
  console.log(`╚══════════════════════════════════════╝\n`);
  await connectEEGSource();
});

process.on('SIGINT', async () => {
  console.log('\n[Server] Shutting down...');
  if (syntheticStream) syntheticStream.stop();
  if (cortex)          await cortex.disconnect();
  process.exit(0);
});
