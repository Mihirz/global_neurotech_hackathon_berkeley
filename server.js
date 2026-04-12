import { config } from 'dotenv';
config();
import express from 'express';
import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import { CortexClient } from './cortex.js';

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
let simInterval      = null;

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
// Mimics Insight format: 5ch, 128Hz, batches of 4 samples
let simTime        = Date.now();
let simFrameOnsets = [];

function startSimulatedEEG() {
  broadcastStatus({ state: 'demo', message: 'Simulated EEG — 128Hz · 5ch (Insight format)' });
  const RATE       = 128;
  const BATCH      = 4;
  const PACKET_MS  = (BATCH / RATE) * 1000; // ~31.25ms

  simInterval = setInterval(() => {
    const packetStart = simTime;
    simTime += PACKET_MS;
    const data = [];

    for (let i = 0; i < BATCH; i++) {
      const t     = packetStart + (i / RATE) * 1000;
      const noise = () => (Math.random() - 0.5) * 4;

      // N170-like deflection: negative on T7(1) and T8(3), positive on Pz(2)
      let n170 = 0, vpp = 0;
      for (const onset of simFrameOnsets) {
        const dt = t - onset;
        if (dt >= 120 && dt <= 220) {
          n170 = -5 * Math.exp(-Math.pow(dt - 170, 2) / (2 * 25 * 25)); // negative
          vpp  =  3 * Math.exp(-Math.pow(dt - 180, 2) / (2 * 20 * 20)); // positive
        }
      }

      // [AF3, T7, Pz, T8, AF4]
      data.push([
        noise(),           // AF3  — frontal, minimal N170
        noise() + n170,    // T7   — left temporal, primary N170
        noise() + vpp,     // Pz   — parietal midline, VPP
        noise() + n170,    // T8   — right temporal, primary N170
        noise(),           // AF4  — frontal, minimal N170
      ]);
    }

    broadcast('eeg_packet', {
      data,
      startTime:  packetStart,
      sampleRate: RATE,
      channels:   ['AF3', 'T7', 'Pz', 'T8', 'AF4'],
      simulated:  true,
    });

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
      droneUrl:        process.env.DRONE_STREAM_URL || null,
      demo:            !process.env.EMOTIV_CLIENT_ID || process.env.EMOTIV_CLIENT_ID === 'your-client-id',
      hasAnthropicKey: !!process.env.ANTHROPIC_API_KEY,
      headset:         'Emotiv Insight',
    },
  }));

  // Replay last known status so browser doesn't show stale white dot
  if (lastCrownStatus) {
    ws.send(JSON.stringify({ type: 'crown_status', payload: lastCrownStatus, ts: Date.now() }));
  }
});

// ── REST: Config ──────────────────────────────────────────────────────────────
app.get('/api/config', (req, res) => {
  res.json({
    droneUrl:        process.env.DRONE_STREAM_URL || null,
    demo:            !process.env.EMOTIV_CLIENT_ID || process.env.EMOTIV_CLIENT_ID === 'your-client-id',
    hasAnthropicKey: !!process.env.ANTHROPIC_API_KEY,
    headset:         'Emotiv Insight',
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
        'Content-Type':      'application/json',
        'x-api-key':         process.env.ANTHROPIC_API_KEY,
        'anthropic-version': '2023-06-01',
      },
      body: JSON.stringify({
        model:      'claude-sonnet-4-20250514',
        max_tokens: 400,
        messages: [{
          role: 'user',
          content: [
            { type: 'image', source: { type: 'base64', media_type: 'image/jpeg', data: base64 } },
            { type: 'text',  text: `Firefighting SAR drone frame. Respond ONLY with valid JSON:
{"personDetected":boolean,"confidence":number,"objects":[],"framePosition":"top-left|top-center|top-right|center-left|center|center-right|bottom-left|bottom-center|bottom-right","estimatedCount":number,"thermalSignature":"string","notes":"string"}` },
          ],
        }],
      }),
    });
    const data = await response.json();
    const text = data.content?.map(b => b.text || '').join('') || '{}';
    let cv;
    try   { cv = JSON.parse(text.replace(/```json|```/g, '').trim()); }
    catch { cv = { personDetected: false, confidence: 0, notes: text, parseError: true }; }
    det.cv = cv;
    console.log(`[CV] Frame ${frameId}: person=${cv.personDetected}, confidence=${cv.confidence}`);
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
  if (simInterval) clearInterval(simInterval);
  if (cortex)      await cortex.disconnect();
  process.exit(0);
});
