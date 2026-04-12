# NeuroRSVP — Firefighter BCI

P300-based image-response collection for firefighter salience detection using a
Muse 2 EEG headset. The collection app presents labeled fire-scene images,
timestamps each stimulus, records concurrent EEG, extracts P300/bandpower
features, and saves rows for training a classifier.

---

## Architecture

```
Drone (MJPEG/WebRTC)
        │
        ▼
  Muse 2 → muselsl stream → muse_lsl_bridge.py
        │                      │
        │ HTTP packets          ▼
        ▼                   [server.js]
  [collection.html + collection.js]  ← WebSocket eeg_packet
        │
        ├── fixation / image / blank trial runner
        ├── P300 epoch extractor (-200ms to +800ms)
        ├── delta/theta/alpha/beta/gamma feature extraction
        └── JSON/CSV session export for model training
```

---

## Quick Start

### 1. Install web dependencies

```bash
npm install
```

### 2. Install Muse bridge dependencies

```bash
pip install -r muse_requirements.txt
```

### 3. Configure `.env`

```bash
cp .env.example .env
```

Edit `.env`:

```env
EEG_SOURCE=muse
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key        # optional fallback/provider
CV_ANALYSIS_PROVIDER=auto                       # auto|openai|anthropic
DRONE_STREAM_URL=http://192.168.1.1:8080/video  # leave blank for demo mode
```

OpenAI is used first when `OPENAI_API_KEY` is set. Anthropic remains available as a fallback or can be forced with `CV_ANALYSIS_PROVIDER=anthropic`.

### 4. Start the Muse stream

Run these in separate terminals:

```bash
muselsl stream
```

```bash
python muse_lsl_bridge.py
```

### 5. Run the web app

```bash
npm start
```

Open **http://localhost:3000/collection** in Chrome or Edge.

---

## Demo Mode

Set `EEG_SOURCE=demo` in `.env` to test without a headset:
- EEG is synthetically generated at 128Hz
- P300 deflections (~6µV) are injected ~350ms after "person" frames
- Thermal-style drone frames are generated with simulated person heat signatures

Demo mode is fully functional for testing the pipeline and UI.

---

## EEG Image Collection App

Open **http://localhost:3000/collection** to run the web-based data collection
station for the sorted image folders:

- `images/humans in fire` -> `human_fire`
- `images/items in fire` -> `item_fire`
- `images/abnormal items` -> `normal`

The collection app presents fixation, image, and blank periods using the PRD
oddball mix (20% humans in fire, 30% items in fire, 50% normal items). It
timestamps every stimulus, extracts a -200ms to +800ms EEG epoch from the live
WebSocket stream, scores the 300-600ms P300 window, and records one row per
image response.

Use **Save to Server** to write the session JSON under `collection_data/`, or
download JSON/CSV directly from the browser. In demo mode, synthetic EEG is
scaled so humans in fire create the strongest P300 response, items in fire create
a moderate response, and normal items create little or no P300 response.

Train a starter classifier from saved sessions:

```bash
python train_muse_p300_classifier.py --sessions "collection_data/*.json"
```

---

## Workflow

### Step 1: Connect Muse 2
Start `muselsl stream`, then start `python muse_lsl_bridge.py`. The collection
page should show `Muse 2 streaming` once packets reach the Node server.

### Step 2: Collect Image Responses
Open `/collection`, set the trial count and timing, then click **Start
Collection**. Each trial shows a fixation cross, one labeled image, then a blank
period while the app waits for the post-stimulus EEG window.

**What to do:** Watch passively and keep still. Do not try to manually classify
each image; the goal is to record the EEG response aligned to each stimulus.

### Step 3: Save and Train
Use **Save to Server** or **Download JSON/CSV** after collection. The saved JSON
contains the label, image ID, stimulus timestamp, P300 features, Muse bandpower
features, channel names, and rejection reason if an epoch was noisy.

Train the starter model with `python train_muse_p300_classifier.py --sessions
"collection_data/*.json"` after collecting enough trials.

---

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Total trials | 90 | Uses 20% human-fire, 30% item-fire, 50% normal |
| Fixation | 500 ms | Pre-image fixation cross |
| Image display | 400 ms | Stimulus display window |
| Blank / EEG window | 900 ms | Lets the +800ms epoch arrive before scoring |
| P300 threshold | 3.0 µV | Initial single-trial threshold |
| Artifact rejection | ±100 µV | Epochs with any channel exceeding this are discarded |
| P300 window | 300–600 ms | Fixed; covers the PRD P300 feature window |
| Baseline | 200 ms pre-stimulus | Used for baseline correction |

---

## Muse 2 Channel Map

```
Typical Muse 2 LSL channels:
  TP9
  AF7
  AF8
  TP10
```

Muse 2 does not have a true Pz electrode. The app uses the available Muse
channels for baseline-corrected P300 and bandpower features, then trains the
classifier on the subject's collected labels. Treat it as a hackathon salience
demo rather than a clinical P300 device.

---

## Drone Feed Setup

### DJI drones (Phantom, Mavic, etc.)
```env
DRONE_STREAM_URL=http://192.168.1.1:8080/video
```
Connect laptop to the drone's WiFi hotspot before starting.

### Parrot drones
```env
DRONE_STREAM_URL=rtsp://192.168.42.1/live
```
Note: rtsp requires a proxy server (ffmpeg → MJPEG). See ffmpeg command below.

### Any drone via ffmpeg bridge
```bash
ffmpeg -i rtsp://your-drone/stream \
  -c:v mjpeg -q:v 5 \
  -f mjpeg http://localhost:8090/
```
Then set `DRONE_STREAM_URL=http://localhost:8090/`

### Pre-recorded video (hackathon fallback)
Place a video file at `public/demo.mp4` and set:
```env
DRONE_STREAM_URL=/demo.mp4
```
This plays the video in the hidden `<video>` element and samples it at RSVP rate.

---

## Known Limitations & Hackathon Workarounds

**Muse 2 must be streaming through LSL**
Start `muselsl stream` before `python muse_lsl_bridge.py`. If the browser says
it is waiting for Muse, the Node server is up but no bridge packets have arrived.

**Artifact rejection may discard too many epochs**
If the EEG plot shows large swings (>100µV), the subject is moving. Reduce the
threshold in `collection.js` (`ARTIFACT_UV`) to 50, or ask the subject to relax.

**P300 latency varies per person**
Default window is 300–600ms. If collected rows show a consistent peak outside
this range, adjust `P300_START_MS` and `P300_END_MS` in `collection.js`.

**Single-trial P300 is noisy**
The system records single-trial rows. For higher accuracy, collect multiple
sessions per user and train on the saved JSON rows instead of trusting one raw
threshold score.

---

## File Structure

```
neurorsvp/
├── server.js                    # Express + WebSocket + Muse bridge endpoints
├── muse_lsl_bridge.py           # Muse 2 LSL -> Node EEG packet bridge
├── collection.html              # Image/EEG collection UI
├── collection.js                # Trial runner, P300 features, band features
├── train_muse_p300_classifier.py # Starter classifier training script
├── index.html                   # Existing drone scan UI
├── app.js                       # Existing RSVP scan UI logic
├── .env.example       # Config template
├── package.json
└── README.md
```

---

## References

- Sajda et al. (2010) "In a blink of an eye and a switch of a transistor" — foundational RSVP-BCI paper
- DARPA Cortically Coupled Computer Vision (C3Vision) program
- Muse LSL tooling: https://github.com/alexandrebarachant/muse-lsl
- MNE-Python P300 analysis: https://mne.tools/stable/auto_examples/
