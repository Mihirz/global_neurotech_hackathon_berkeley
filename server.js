import { config } from 'dotenv';
config();
import express from 'express';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import { Neurosity } from '@neurosity/sdk';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const server = createServer(app);
const wss = new WebSocketServer({ server });

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json({ limit: '10mb' }));

// ── Session store ─────────────────────────────────────────────────────────────
const sessionDetections = [];
const telemetryLog = [];
let latestTelemetry = null;

let neurosity = null;
let eegSubscription = null;
let statusSubscription = null;
let connectedClients = new Set();
let lastCrownStatus = null; // stored so late-connecting clients get current state

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

// ── Neurosity Crown ───────────────────────────────────────────────────────────
async function connectCrown() {
  if (!process.env.NEUROSITY_EMAIL || process.env.NEUROSITY_EMAIL === 'your@email.com') {
    console.log('[Crown] No credentials — running in DEMO mode');
    startSimulatedEEG();
    return;
  }
  try {
    neurosity = new Neurosity({ deviceId: process.env.NEUROSITY_DEVICE_ID });
    await neurosity.login({ email: process.env.NEUROSITY_EMAIL, password: process.env.NEUROSITY_PASSWORD });
    console.log('[Crown] Logged in');
    statusSubscription = neurosity.status().subscribe((status) => {
      console.log('[Crown] Status:', status.state);
      broadcastStatus(status);
    });
    await neurosity.selectDevice([process.env.NEUROSITY_DEVICE_ID]);
    eegSubscription = neurosity.brainwaves('rawUnfiltered').subscribe((brainwave) => {
      broadcast('eeg_packet', {
        data: brainwave.data,
        startTime: brainwave.info.startTime,
        sampleRate: 256,
        channels: ['CP3','C3','F5','PO3','PO4','F6','C4','CP4'],
      });
    });
    console.log('[Crown] EEG streaming started');
    broadcastStatus({ state: 'online', message: 'Crown connected — EEG streaming' });
  } catch (err) {
    console.error('[Crown] Connection failed:', err.message);
    broadcastStatus({ state: 'error', message: err.message });
    startSimulatedEEG();
  }
}

// ── Simulated EEG ─────────────────────────────────────────────────────────────
let simInterval = null;
let simTime = Date.now();
let simFrameOnsets = [];

function startSimulatedEEG() {
  broadcastStatus({ state: 'demo', message: 'Simulated EEG — 256Hz, 8ch' });
  const RATE = 256, PACKET_SIZE = 16;
  const PACKET_MS = (PACKET_SIZE / RATE) * 1000;
  simInterval = setInterval(() => {
    const packetStart = simTime;
    simTime += PACKET_MS;
    const data = [];
    for (let i = 0; i < PACKET_SIZE; i++) {
      const t = packetStart + (i / RATE) * 1000;
      const noise = () => (Math.random() - 0.5) * 4;
      let p300 = 0;
      for (const onset of simFrameOnsets) {
        const dt = t - onset;
        if (dt >= 250 && dt <= 550) p300 = 6 * Math.exp(-Math.pow(dt - 350, 2) / (2 * 80 * 80));
      }
      data.push([noise(), noise(), noise(), noise() + p300, noise() + p300, noise(), noise(), noise()]);
    }
    broadcast('eeg_packet', { data, startTime: packetStart, sampleRate: RATE, channels: ['CP3','C3','F5','PO3','PO4','F6','C4','CP4'], simulated: true });
    simFrameOnsets = simFrameOnsets.filter(o => o > Date.now() - 3000);
  }, PACKET_MS);
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
wss.on('connection', (ws) => {
  connectedClients.add(ws);
  console.log(`[WS] Client connected (${connectedClients.size} total)`);
  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw);
      if (msg.type === 'sim_frame_onset') simFrameOnsets.push(msg.ts);
      if (msg.type === 'ping') ws.send(JSON.stringify({ type: 'pong', ts: Date.now() }));
    } catch {}
  });
  ws.on('close', () => {
    connectedClients.delete(ws);
    console.log(`[WS] Client disconnected (${connectedClients.size} total)`);
  });
  ws.send(JSON.stringify({
    type: 'init',
    payload: {
      droneUrl: process.env.DRONE_STREAM_URL || null,
      demo: !process.env.NEUROSITY_EMAIL || process.env.NEUROSITY_EMAIL === 'your@email.com',
      hasAnthropicKey: !!process.env.ANTHROPIC_API_KEY,
    }
  }));

  // Replay last known Crown status so the client doesn't show stale/white indicator
  if (lastCrownStatus) {
    ws.send(JSON.stringify({ type: 'crown_status', payload: lastCrownStatus, ts: Date.now() }));
  }
});

// ── REST: Config ──────────────────────────────────────────────────────────────
app.get('/api/config', (req, res) => {
  res.json({
    droneUrl: process.env.DRONE_STREAM_URL || null,
    demo: !process.env.NEUROSITY_EMAIL || process.env.NEUROSITY_EMAIL === 'your@email.com',
    hasAnthropicKey: !!process.env.ANTHROPIC_API_KEY,
  });
});

// ── REST: Detections ──────────────────────────────────────────────────────────
app.post('/api/detections', (req, res) => {
  const { frameId, ts, imgSrc, amplitude, score, gps } = req.body;
  if (!sessionDetections.find(d => d.frameId === frameId)) {
    sessionDetections.push({
      frameId, ts, imgSrc, amplitude, score,
      gps: gps || latestTelemetry || null,
      cv: null, reviewed: false, confirmed: false,
    });
    console.log(`[Session] Detection logged: frame ${frameId} (${amplitude}µV)`);
  }
  res.json({ ok: true, total: sessionDetections.length });
});

app.get('/api/detections', (req, res) => res.json(sessionDetections));

app.patch('/api/detections/:frameId', (req, res) => {
  const frameId = parseInt(req.params.frameId);
  const det = sessionDetections.find(d => d.frameId === frameId);
  if (!det) return res.status(404).json({ error: 'Not found' });
  Object.assign(det, req.body);
  broadcast('detection_updated', det);
  res.json(det);
});

app.delete('/api/detections', (req, res) => {
  sessionDetections.length = 0;
  res.json({ ok: true });
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

// ── REST: CV Analysis (Anthropic Vision) ──────────────────────────────────────
app.post('/api/analyze/:frameId', async (req, res) => {
  const frameId = parseInt(req.params.frameId);
  const det = sessionDetections.find(d => d.frameId === frameId);
  if (!det) return res.status(404).json({ error: 'Detection not found' });

  if (!process.env.ANTHROPIC_API_KEY) {
    const demo = {
      personDetected: true, confidence: 0.87,
      objects: ['person', 'possible victim'],
      framePosition: 'center-left', estimatedCount: 1,
      thermalSignature: 'high — consistent with living person',
      notes: 'Demo mode — set ANTHROPIC_API_KEY in .env for real analysis',
      demo: true,
    };
    det.cv = demo;
    return res.json(demo);
  }

  try {
    const base64 = det.imgSrc.replace(/^data:image\/\w+;base64,/, '');
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 400,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: base64 } },
            { type: 'text', text: `You are analyzing a thermal/visual drone frame from a firefighting search and rescue operation. Respond ONLY with valid JSON, no markdown, no explanation:
{"personDetected":boolean,"confidence":number,"objects":[],"framePosition":"one of top-left/top-center/top-right/center-left/center/center-right/bottom-left/bottom-center/bottom-right","estimatedCount":number,"thermalSignature":"string","notes":"string"}` }
          ]
        }]
      })
    });
    const data = await response.json();
    const text = data.content?.map(b => b.text || '').join('') || '{}';
    let cv;
    try { cv = JSON.parse(text.replace(/```json|```/g, '').trim()); }
    catch { cv = { personDetected: false, confidence: 0, notes: text, parseError: true }; }
    det.cv = cv;
    console.log(`[CV] Frame ${frameId}: person=${cv.personDetected}, confidence=${cv.confidence}`);
    res.json(cv);
  } catch (err) {
    console.error('[CV] Analysis failed:', err.message);
    res.status(500).json({ error: err.message });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
server.listen(PORT, async () => {
  console.log(`\n╔══════════════════════════════════════╗`);
  console.log(`║   NeuroRSVP — Firefighter BCI       ║`);
  console.log(`║   http://localhost:${PORT}               ║`);
  console.log(`║   Review: http://localhost:${PORT}/review║`);
  console.log(`╚══════════════════════════════════════╝\n`);
  await connectCrown();
});

process.on('SIGINT', async () => {
  console.log('\n[Server] Shutting down...');
  if (eegSubscription) eegSubscription.unsubscribe();
  if (statusSubscription) statusSubscription.unsubscribe();
  if (simInterval) clearInterval(simInterval);
  if (neurosity) await neurosity.logout();
  process.exit(0);
});