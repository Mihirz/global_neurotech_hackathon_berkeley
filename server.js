import 'dotenv/config';
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
app.use(express.json());

// --- Neurosity Crown setup ---
let neurosity = null;
let eegSubscription = null;
let statusSubscription = null;
let connectedClients = new Set();

function broadcast(type, payload) {
  const msg = JSON.stringify({ type, payload, ts: Date.now() });
  for (const client of connectedClients) {
    if (client.readyState === 1) client.send(msg);
  }
}

async function connectCrown() {
  if (!process.env.NEUROSITY_EMAIL || process.env.NEUROSITY_EMAIL === 'your@email.com') {
    console.log('[Crown] No credentials — running in DEMO mode (simulated EEG)');
    broadcast('crown_status', { state: 'demo', message: 'Demo mode — simulated EEG active' });
    startSimulatedEEG();
    return;
  }

  try {
    neurosity = new Neurosity({ deviceId: process.env.NEUROSITY_DEVICE_ID });

    await neurosity.login({
      email: process.env.NEUROSITY_EMAIL,
      password: process.env.NEUROSITY_PASSWORD,
    });

    console.log('[Crown] Logged in');

    // Monitor device status
    statusSubscription = neurosity.status().subscribe((status) => {
      console.log('[Crown] Status:', status.state);
      broadcast('crown_status', status);
    });

    // Select device and wait for online state
    await neurosity.selectDevice([process.env.NEUROSITY_DEVICE_ID]);

    // Stream raw EEG — 256Hz, 8 channels, 16 samples per packet
    eegSubscription = neurosity.brainwaves('raw').subscribe((brainwave) => {
      broadcast('eeg_packet', {
        data: brainwave.data,       // array of sample arrays: [[ch0..ch7], ...]
        startTime: brainwave.info.startTime,
        sampleRate: 256,
        channels: ['CP3','C3','F5','PO3','PO4','F6','C4','CP4'],
      });
    });

    console.log('[Crown] EEG streaming started');
    broadcast('crown_status', { state: 'online', message: 'Crown connected — EEG streaming' });

  } catch (err) {
    console.error('[Crown] Connection failed:', err.message);
    broadcast('crown_status', { state: 'error', message: err.message });
    console.log('[Crown] Falling back to demo mode');
    startSimulatedEEG();
  }
}

// Demo mode: simulate 256Hz EEG with occasional P300-like deflections
let simInterval = null;
let simTime = Date.now();
let simFrameOnsets = []; // shared with frontend via broadcast

function startSimulatedEEG() {
  broadcast('crown_status', { state: 'demo', message: 'Simulated EEG — 256Hz, 8ch' });

  const RATE = 256;
  const PACKET_SIZE = 16;
  const PACKET_MS = (PACKET_SIZE / RATE) * 1000; // ~62.5ms

  simInterval = setInterval(() => {
    const packetStart = simTime;
    simTime += PACKET_MS;

    const data = [];
    for (let i = 0; i < PACKET_SIZE; i++) {
      const t = packetStart + (i / RATE) * 1000;
      // Base noise: pink noise approximation
      const noise = () => (Math.random() - 0.5) * 4;
      // Simulate a P300 at ~350ms after a flagged stimulus onset
      let p300 = 0;
      for (const onset of simFrameOnsets) {
        const dt = t - onset;
        if (dt >= 250 && dt <= 550) {
          // Gaussian bump centered at 350ms, amplitude 6µV on parietal channels
          p300 = 6 * Math.exp(-Math.pow(dt - 350, 2) / (2 * 80 * 80));
        }
      }
      data.push([
        noise(), noise(), noise(),
        noise() + p300,  // PO3 — index 3
        noise() + p300,  // PO4 — index 4
        noise(), noise(), noise(),
      ]);
    }

    broadcast('eeg_packet', {
      data,
      startTime: packetStart,
      sampleRate: RATE,
      channels: ['CP3','C3','F5','PO3','PO4','F6','C4','CP4'],
      simulated: true,
    });

    // Prune old onsets
    const cutoff = Date.now() - 3000;
    simFrameOnsets = simFrameOnsets.filter(o => o > cutoff);
  }, PACKET_MS);
}

// WebSocket connection handler
wss.on('connection', (ws) => {
  connectedClients.add(ws);
  console.log(`[WS] Client connected (${connectedClients.size} total)`);

  ws.on('message', (raw) => {
    try {
      const msg = JSON.parse(raw);

      if (msg.type === 'sim_frame_onset') {
        // Frontend tells us a "target" frame was shown (for demo P300 injection)
        simFrameOnsets.push(msg.ts);
      }

      if (msg.type === 'ping') {
        ws.send(JSON.stringify({ type: 'pong', ts: Date.now() }));
      }
    } catch (e) {
      // ignore
    }
  });

  ws.on('close', () => {
    connectedClients.delete(ws);
    console.log(`[WS] Client disconnected (${connectedClients.size} total)`);
  });

  // Send current state on connect
  ws.send(JSON.stringify({
    type: 'init',
    payload: {
      droneUrl: process.env.DRONE_STREAM_URL || null,
      demo: !process.env.NEUROSITY_EMAIL || process.env.NEUROSITY_EMAIL === 'your@email.com',
    }
  }));
});

// REST: get server config
app.get('/api/config', (req, res) => {
  res.json({
    droneUrl: process.env.DRONE_STREAM_URL || null,
    demo: !process.env.NEUROSITY_EMAIL || process.env.NEUROSITY_EMAIL === 'your@email.com',
  });
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, async () => {
  console.log(`\n╔══════════════════════════════════════╗`);
  console.log(`║   NeuroRSVP — Firefighter BCI       ║`);
  console.log(`║   http://localhost:${PORT}               ║`);
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
