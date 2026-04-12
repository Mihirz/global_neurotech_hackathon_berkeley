# NeuroRSVP — Firefighter BCI

CV-led image-response collection for firefighter salience detection using a
live Muse 2 EEG headset overlay. The collection app presents labeled fire-scene
images, timestamps each stimulus, records concurrent EEG, extracts raw
P300-window/bandpower features, and saves rows for analysis. Classifier training
is bypassed for the hackathon UI: decisions follow the image/CV diagnosis, with
live Muse bands and raw epoch features adding a small salience bias.

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
        ├── Muse epoch extractor (-200ms to +800ms)
        ├── delta/theta/alpha/beta/gamma feature extraction
        └── CV-led salience scoring + JSON/CSV session export
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
- Synthetic EEG is generated at the same endpoint shape as the Muse bridge
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
WebSocket stream, computes raw 300-600ms P300-window and bandpower fields, and
records one row per image response. The displayed decision is CV-led: humans in
fire are high salience, items in fire are medium salience, and normal items are
low salience, with a small live Muse/random bias layered in.

Use **Save to Server** to write the session JSON under `collection_data/`. Saving
does not start classifier training; the web UI stays in CV-led Muse mode. Use
**Download JSON/CSV** directly from the browser when you want the rows. The old
trainer is still included as a research utility, but it is not part of the main
demo path.

Optional research-only trainer:

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

### Step 3: Save or Export
Use **Save to Server** or **Download JSON/CSV** after collection. The saved JSON
contains the label, CV diagnosis, image ID, stimulus timestamp, salience score,
raw P300-window features, Muse bandpower features, channel names, and signal
quality warnings. No rows are rejected and no automatic training job is started.

---

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Total trials | 90 | Uses 20% human-fire, 30% item-fire, 50% normal |
| Fixation | 500 ms | Pre-image fixation cross |
| Image display | 400 ms | Stimulus display window |
| Blank / EEG window | 900 ms | Lets the +800ms epoch arrive before scoring |
| Raw Muse marker threshold | 3.0 µV | Used only for exported raw P300-window fields |
| Artifact warning | ±100 µV | High-amplitude epochs are marked noisy but still kept |
| P300 window | 300–600 ms | Fixed raw feature window; not the primary decision source |
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
channels for baseline-corrected raw P300-window and bandpower features, while
the UI decision is led by the image/CV diagnosis. Treat it as a hackathon
salience demo rather than a clinical P300 device.

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

**Artifact warnings may appear**
If the EEG plot rails near ±1000µV, the Muse electrodes are saturating. Re-wet
or reseat the Muse sensors and keep the subject still. The app records these
rows with `signalQuality` warnings instead of rejecting them, but noisy rows are
still lower quality EEG evidence.

**P300 latency varies per person**
Default window is 300–600ms. If collected rows show a consistent peak outside
this range, adjust `P300_START_MS` and `P300_END_MS` in `collection.js`.

**Single-trial P300 is noisy**
The system records single-trial rows, but the hackathon UI no longer trusts a
raw threshold score for the verdict. It uses CV-led salience with the live Muse
stream visible and stored alongside each image response.

---

## File Structure

```
neurorsvp/
├── server.js                    # Express + WebSocket + Muse bridge endpoints
├── muse_lsl_bridge.py           # Muse 2 LSL -> Node EEG packet bridge
├── collection.html              # Image/EEG collection UI
├── collection.js                # Trial runner, CV-led Muse salience, raw features
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
