import { config } from 'dotenv';
config();
import express from 'express';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
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

// ── Session store ─────────────────────────────────────────────────────────────
const sessionDetections = [];
const telemetryLog      = [];
let   latestTelemetry   = null;

let connectedClients = new Set();
let lastCrownStatus  = null;
let cortex           = null;

const CV_PROMPT = `Firefighting SAR drone frame. Respond ONLY with valid JSON:
{"personDetected":boolean,"confidence":number,"objects":[],"framePosition":"top-left|top-center|top-right|center-left|center|center-right|bottom-left|bottom-center|bottom-right","estimatedCount":number,"thermalSignature":"string","notes":"string"}`;

function getCvProvider() {
  const requested = (process.env.CV_ANALYSIS_PROVIDER || 'auto').toLowerCase();
  if (requested === 'openai' && process.env.OPENAI_API_KEY) return 'openai';
  if (requested === 'anthropic' && process.env.ANTHROPIC_API_KEY) return 'anthropic';
  if (process.env.OPENAI_API_KEY) return 'openai';
  if (process.env.ANTHROPIC_API_KEY) return 'anthropic';
  return null;
}

function getConfigPayload() {
  const cvProvider = getCvProvider();
  return {
    droneUrl:         process.env.DRONE_STREAM_URL || null,
    demo:             !process.env.EMOTIV_CLIENT_ID || process.env.EMOTIV_CLIENT_ID === 'your-client-id',
    hasAnthropicKey:  !!process.env.ANTHROPIC_API_KEY,
    hasOpenAIKey:     !!process.env.OPENAI_API_KEY,
    hasAiAnalysisKey: !!cvProvider,
    cvProvider:       cvProvider || 'demo',
    headset:          'Emotiv Insight',
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
        syntheticStream.notifyFrameOnset(msg.ts, isTarget);
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

// ── Start ─────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
server.listen(PORT, async () => {
  console.log(`\n╔══════════════════════════════════════╗`);
  console.log(`║   NeuroRSVP — Emotiv Insight BCI    ║`);
  console.log(`║   http://localhost:${PORT}               ║`);
  console.log(`║   Review: http://localhost:${PORT}/review║`);
  console.log(`╚══════════════════════════════════════╝\n`);
  await connectInsight();
});

process.on('SIGINT', async () => {
  console.log('\n[Server] Shutting down...');
  if (syntheticStream) syntheticStream.stop();
  if (cortex)          await cortex.disconnect();
  process.exit(0);
});
