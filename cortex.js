/**
 * cortex.js — Emotiv Cortex API client for the Insight headset
 *
 * Connects to the EMOTIV Launcher's local Cortex service at wss://localhost:6868
 * Handles the full auth flow: requestAccess → authorize → queryHeadsets →
 * createSession → subscribe to EEG stream.
 *
 * Emits callbacks:
 *   onStatus(message)              — human-readable status string
 *   onEEGPacket({data, startTime, sampleRate, channels})
 *                                  — same shape as Neurosity packets for compatibility
 *
 * Insight channel order in EEG packets (indices 2–6 after COUNTER, INTERPOLATED):
 *   0: AF3   1: T7   2: Pz   3: T8   4: AF4
 */

import WebSocket from 'ws';

const CORTEX_URL   = 'wss://localhost:6868';
const SAMPLE_RATE  = 128;
const CHANNELS     = ['AF3', 'T7', 'Pz', 'T8', 'AF4'];

// Indices of EEG values in the raw 'eeg' array from Cortex:
// [COUNTER, INTERPOLATED, AF3, T7, Pz, T8, AF4, RAW_CQ, MARKER_HARDWARE, MARKERS]
const EEG_START_IDX = 2;
const EEG_N_CH      = 5;

export class CortexClient {
  constructor({ clientId, clientSecret, onStatus, onEEGPacket }) {
    this.clientId     = clientId;
    this.clientSecret = clientSecret;
    this.onStatus     = onStatus    || (() => {});
    this.onEEGPacket  = onEEGPacket || (() => {});

    this.ws           = null;
    this.msgId        = 1;
    this.pending      = new Map();   // id → {resolve, reject}
    this.cortexToken  = null;
    this.sessionId    = null;
    this.headsetId    = null;

    // Packet accumulator — Cortex sends one sample at a time;
    // we batch 4 samples (~31ms) to match the upstream format
    this.sampleBatch    = [];
    this.batchStartTime = null;
    this.BATCH_SIZE     = 4;
  }

  // ── Connect and run full auth flow ─────────────────────────────────────────
  async connect() {
    this.onStatus('Connecting to EMOTIV Launcher (wss://localhost:6868)...');

    await new Promise((resolve, reject) => {
      this.ws = new WebSocket(CORTEX_URL, { rejectUnauthorized: false });

      this.ws.on('open', resolve);
      this.ws.on('error', (err) => {
        reject(new Error(
          'Cannot reach Cortex service. Make sure EMOTIV Launcher is running. ' + err.message
        ));
      });
    });

    this.ws.on('message', (raw) => this._onMessage(raw));
    this.ws.on('close',   ()    => this.onStatus('Cortex connection closed'));

    this.onStatus('Cortex connected — requesting access...');

    // Step 1: Request access (opens Emotiv Launcher approval dialog first time)
    const access = await this._call('requestAccess', {
      clientId: this.clientId,
      clientSecret: this.clientSecret,
    });
    if (!access.accessGranted) {
      throw new Error(
        'Access not granted. Open EMOTIV Launcher and approve this app, then restart.'
      );
    }

    // Step 2: Authorize — get cortexToken
    this.onStatus('Access granted — authorizing...');
    const auth = await this._call('authorize', {
      clientId:     this.clientId,
      clientSecret: this.clientSecret,
      debit:        1,
    });
    this.cortexToken = auth.cortexToken;
    this.onStatus('Authorized — searching for Insight headset...');

    // Step 3: Find headset
    await this._connectHeadset();

    // Step 4: Create session
    this.onStatus('Creating session...');
    const session = await this._call('createSession', {
      cortexToken: this.cortexToken,
      headset:     this.headsetId,
      status:      'open',
    });
    this.sessionId = session.id;
    this.onStatus(`Session open (${this.sessionId.slice(0, 8)}...) — subscribing to EEG...`);

    // Step 5: Subscribe to EEG stream
    await this._call('subscribe', {
      cortexToken: this.cortexToken,
      session:     this.sessionId,
      streams:     ['eeg'],
    });

    this.onStatus('Emotiv Insight streaming — 128Hz, 5ch (AF3 T7 Pz T8 AF4)');
  }

  // ── Find and connect the Insight headset ───────────────────────────────────
  async _connectHeadset(attempt = 0) {
    const headsets = await this._call('queryHeadsets', {});

    const insight = headsets.find(h => h.id.startsWith('INSIGHT'));
    if (!insight) {
      if (attempt >= 6) throw new Error('No Insight headset found after 12 seconds. Check USB dongle.');
      this.onStatus(`No headset found — retrying (${attempt + 1}/6)...`);
      await sleep(2000);
      return this._connectHeadset(attempt + 1);
    }

    this.headsetId = insight.id;

    if (insight.status !== 'connected') {
      this.onStatus(`Found ${insight.id} (${insight.status}) — connecting...`);
      await this._call('controlDevice', {
        command: 'connect',
        headset: this.headsetId,
      });
      // Wait for connection
      await sleep(3000);
    } else {
      this.onStatus(`Found ${insight.id} — already connected`);
    }
  }

  // ── Close session and disconnect ───────────────────────────────────────────
  async disconnect() {
    if (this.sessionId && this.cortexToken) {
      try {
        await this._call('updateSession', {
          cortexToken: this.cortexToken,
          session:     this.sessionId,
          status:      'close',
        });
      } catch {}
    }
    if (this.ws) this.ws.close();
  }

  // ── Message routing ────────────────────────────────────────────────────────
  _onMessage(raw) {
    let msg;
    try { msg = JSON.parse(raw); } catch { return; }

    // Resolve pending RPC call
    if (msg.id && this.pending.has(msg.id)) {
      const { resolve, reject } = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      if (msg.error) reject(new Error(`Cortex error ${msg.error.code}: ${msg.error.message}`));
      else           resolve(msg.result);
      return;
    }

    // EEG data stream push: { "eeg": [...], "sid": "...", "time": 1234.56 }
    if (msg.eeg) {
      this._handleEEGSample(msg);
    }
  }

  // ── EEG sample handler ─────────────────────────────────────────────────────
  _handleEEGSample(msg) {
    const raw    = msg.eeg;
    const time   = msg.time * 1000;  // Cortex timestamps are in seconds → ms

    // Extract 5 channel values (indices 2–6)
    const channels = [];
    for (let i = 0; i < EEG_N_CH; i++) {
      channels.push(raw[EEG_START_IDX + i] || 0);
    }

    if (this.batchStartTime === null) this.batchStartTime = time;
    this.sampleBatch.push(channels);

    if (this.sampleBatch.length >= this.BATCH_SIZE) {
      this.onEEGPacket({
        data:       this.sampleBatch,
        startTime:  this.batchStartTime,
        sampleRate: SAMPLE_RATE,
        channels:   CHANNELS,
      });
      this.sampleBatch    = [];
      this.batchStartTime = null;
    }
  }

  // ── JSON-RPC call helper ───────────────────────────────────────────────────
  _call(method, params) {
    return new Promise((resolve, reject) => {
      const id  = this.msgId++;
      const msg = JSON.stringify({ id, jsonrpc: '2.0', method, params });

      this.pending.set(id, { resolve, reject });
      this.ws.send(msg);

      // Timeout after 15s for slow operations (headset connect)
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id);
          reject(new Error(`Cortex RPC timeout: ${method}`));
        }
      }, 15000);
    });
  }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
