# NeuroRSVP — Firefighter BCI

P300-based human detection triage using a Neurosity Crown and drone live feed.
Presents drone frames via Rapid Serial Visual Presentation (RSVP) and detects
P300 ERP components to flag frames where the wearer subconsciously perceived a person.

---

## Architecture

```
Drone (MJPEG/WebRTC)
        │
        ▼
  [server.js]  ←──── Neurosity Crown SDK (256Hz EEG)
        │                      │
        │ WebSocket             │ brainwaves('raw')
        ▼                      ▼
  [index.html + app.js]   (streamed to frontend via WS)
        │
        ├── RSVP engine (8fps frame presenter)
        ├── p300.js (epoch extractor + classifier)
        └── Flagged frame gallery
```

---

## Quick Start

### 1. Install dependencies

```bash
npm install
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```env
NEUROSITY_EMAIL=your@email.com
NEUROSITY_PASSWORD=yourpassword
NEUROSITY_DEVICE_ID=your-device-id          # found in Neurosity Console
DRONE_STREAM_URL=http://192.168.1.1:8080/video  # leave blank for demo mode
```

**Finding your device ID:** Log into [console.neurosity.co](https://console.neurosity.co),
go to Devices → your Crown → copy the Device ID from the URL or device details.

### 3. Run

```bash
npm start
```

Open **http://localhost:3000** in Chrome or Edge.

---

## Demo Mode

If no Neurosity credentials are set, the app runs in **demo mode**:
- EEG is synthetically generated at 256Hz
- P300 deflections (~6µV) are injected ~350ms after "person" frames
- Thermal-style drone frames are generated with simulated person heat signatures

Demo mode is fully functional for testing the pipeline and UI.

---

## Workflow

### Step 1: Calibrate
Click **Calibrate Crown** before scanning.

The calibration routine presents 75 images (15 targets with simulated persons,
60 non-targets without) at 400ms ISI. It computes the grand-average P300
component and sets a personalized detection threshold.

**What to do:** Watch the calibration images and mentally note when you see a person.
Do not blink excessively. Remain still.

**Duration:** ~30 seconds.

After calibration, you'll see the target vs non-target ERP traces plotted.
The orange trace (target) should show a positive deflection at 300–500ms.

### Step 2: Scan
Click **Begin Scan** to start the RSVP loop.

Drone frames are presented at **8Hz** (125ms each). For each frame:
1. Frame is sampled from the drone feed (or simulated)
2. Presented fullscreen in the RSVP window
3. 600ms later, the EEG epoch is extracted and scored
4. If P300 amplitude > threshold → frame flagged in the Detection Gallery

**What to do:** Watch the RSVP stream. Do not try to consciously judge frames —
your P300 fires pre-consciously. Passive observation gives the best signal.

### Step 3: Review
Flagged frames appear in the right panel, newest first, sorted by detection time.
Frame ID and P300 amplitude are shown on each card.

---

## Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Presentation rate | 8 Hz | 4–16 Hz. Slower = stronger P300 but lower throughput |
| P300 threshold | 3.0 µV | Lowered automatically after calibration |
| Artifact rejection | ±100 µV | Epochs with any channel exceeding this are discarded |
| P300 window | 250–550 ms | Fixed; covers typical P300 latency range |
| Baseline | 100 ms pre-stimulus | Used for baseline correction |

---

## EEG Channel Map (Neurosity Crown)

```
Index  Label   Position        P300 relevance
  0    CP3     Left centroparietal   Secondary
  1    C3      Left central          Low
  2    F5      Left frontal          Low
  3    PO3     Left parieto-occipital  PRIMARY
  4    PO4     Right parieto-occipital PRIMARY
  5    F6      Right frontal         Low
  6    C4      Right central         Low
  7    CP4     Right centroparietal  Secondary
```

P300 is scored from the average of PO3 (index 3) and PO4 (index 4).

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

**Crown needs contact quality check first**
Before calibrating, run the Neurosity app on your phone and confirm all channels
show green contact quality. Poor contact → no signal.

**Artifact rejection may discard too many epochs**
If the EEG plot shows large swings (>100µV), the subject is moving. Reduce the
threshold in `p300.js` → `ARTIFACT_THRESH_UV` to 50, or ask the subject to relax.

**P300 latency varies per person**
Default window is 250–550ms. If calibration shows the peak outside this range,
adjust `P300_WIN_START` and `P300_WIN_END` in `p300.js`.

**Single-trial P300 is noisy**
The system makes single-trial decisions (one epoch per frame). For higher accuracy,
average multiple passes of the same flight path. The `stimulusLog` array stores
all epochs — you can implement post-hoc averaging.

---

## File Structure

```
neurorsvp/
├── server.js          # Express + WebSocket + Neurosity Crown connection
├── public/
│   ├── index.html     # Main UI (tactical dark, Barlow Condensed)
│   ├── app.js         # Frontend: RSVP engine, EEG display, UI logic
│   └── p300.js        # Signal processing: EEGBuffer, epoch extraction, Calibrator
├── .env.example       # Config template
├── package.json
└── README.md
```

---

## References

- Sajda et al. (2010) "In a blink of an eye and a switch of a transistor" — foundational RSVP-BCI paper
- DARPA Cortically Coupled Computer Vision (C3Vision) program
- Neurosity SDK docs: https://docs.neurosity.co/docs/reference/overview
- MNE-Python P300 analysis: https://mne.tools/stable/auto_examples/
