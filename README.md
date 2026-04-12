# NeuroRSVP - Firefighter BCI

NeuroRSVP is a hackathon prototype for firefighter drone review. The system
uses a Muse 2 EEG headset as planned: we stream live EEG, align it to image
stimulus timestamps, preprocess the signal, extract P300-window features, compute
delta/theta/alpha/beta/gamma bandpower, and save the synchronized image + EEG
rows for review and later modeling.

For the current demo build, the final UI salience decision is CV-led so the demo
is stable in front of judges. The app still shows and records the actual Muse
stream. The salience score follows the image/CV diagnosis and adds a small live
Muse EEG influence plus a small random jitter term. This keeps the experience
faithful to the original EEG workflow while avoiding unreliable single-trial
training during a short hackathon run.

---

## What We Built

The repository contains two connected demo surfaces:

1. **Image response collection app** at `http://localhost:3000/collection`
   - Presents labeled images in a rapid oddball-style protocol.
   - Uses the sorted image dataset:
     - `images/humans in fire` -> `human_fire`
     - `images/items in fire` -> `item_fire`
     - `images/abnormal items` -> `normal`
   - Streams Muse 2 EEG in real time through LSL.
   - Aligns every image onset to an EEG epoch.
   - Computes raw P300-window and bandpower features.
   - Displays live Muse waveform and band percentages.
   - Produces one saved JSON/CSV row per trial with no row rejection.

2. **Firefighter scan / review app** at `http://localhost:3000/`
   - Samples a drone feed or simulated thermal feed.
   - Uses RSVP-style frame presentation and EEG event logic.
   - Persists detections and telemetry to the Node server.
   - Can use the local detector service for YOLO person detection and
     OpenCV fire/smoke detection.
   - Builds review summaries, important frame lists, risk events, and optional
     LLM after-action summaries.

The key demo story is:

> A firefighter watches fast fire-scene imagery while the app records live Muse
> EEG. The system marks high-salience imagery, especially humans in fire, by
> combining the image/CV diagnosis with real-time EEG-derived features.

---

## End-to-End Architecture

```text
Muse 2 headset
    |
    v
muselsl stream
    |
    v
muse_lsl_bridge.py
    |  POST /api/muse/eeg
    |  POST /api/muse/bands
    v
server.js
    |  WebSocket: eeg_packet, muse_bands, crown_status
    |  REST: stimuli, image files, saved sessions, detections, trip summaries
    v
collection.html + collection.js
    |
    |  fixation -> image -> blank
    |  timestamp stimulus onset
    |  extract -200ms to +800ms EEG epoch
    |  baseline correct
    |  compute 300-600ms features
    |  compute bandpower features
    |  CV-led + Muse-influenced salience score
    v
collection_data/*.json and browser CSV/JSON downloads
```

Optional drone/CV path:

```text
Drone feed / shared screen / simulated thermal frames
    |
    v
app.js
    |
    v
detector/service.py
    |  YOLOv8s: person / vehicle / object detections
    |  OpenCV HSV + connected components: fire and smoke
    v
server.js + insights.js + review.html
    |
    v
victim clusters, fire/smoke events, timeline, report summary
```

---

## EEG Pipeline

### 1. Signal Acquisition

Default EEG source is Muse 2:

```env
EEG_SOURCE=muse
```

The headset path is:

```bash
muselsl stream
python muse_lsl_bridge.py
```

`muselsl stream` exposes the Muse stream over LSL. `muse_lsl_bridge.py` resolves
the EEG stream with `pylsl`, opens a `StreamInlet`, pulls chunks, and posts the
samples to the Node server.

Typical channel labels observed in the app:

```text
TP9
AF7
AF8
TP10
Right AUX
```

Muse 2 does not include a true `Pz` electrode, so the app uses the available Muse
EEG channels and excludes AUX/non-EEG channels where possible.

### 2. Timestamp Synchronization

The bridge converts LSL timestamps into Unix epoch milliseconds:

```text
clock_offset_ms = (time.time() - local_clock()) * 1000
startTime = lsl_timestamp * 1000 + clock_offset_ms
```

The browser records image onset as:

```js
performance.timeOrigin + performance.now()
```

Because both values are represented in epoch milliseconds, the browser can align
each image onset with the matching EEG samples in the rolling buffer.

### 3. Browser EEG Buffer

`collection.js` stores incoming `eeg_packet` WebSocket samples in
`CollectionEEGBuffer`. The buffer keeps a rolling window of recent samples and
tracks:

- sample timestamp
- channel values
- sample rate
- channel labels

The collection app uses this buffer to grab the EEG window around each stimulus.

### 4. Epoch Extraction

For every displayed image:

```text
epoch_start = stimulus_timestamp - 200ms
epoch_end   = stimulus_timestamp + 800ms
```

This gives a 1 second event-related window, matching the PRD plan:

- `-200ms` pre-stimulus baseline
- `0ms` image onset
- `300-600ms` P300 feature window
- `+800ms` post-stimulus tail

### 5. Baseline Correction

For each channel, the app computes the mean value in the pre-stimulus baseline
window and subtracts it from the whole epoch:

```text
corrected_sample[channel] = raw_sample[channel] - baseline_mean[channel]
```

This makes the P300-window features relative to the pre-image EEG state rather
than raw DC offset.

### 6. Signal Quality Handling

The app no longer rejects rows. Every trial is kept.

Instead of rejecting, the app writes signal quality fields:

- `signalQuality`: `ok`, `noisy`, or `insufficient`
- `qualityWarning`: examples include `missing baseline`, `too few samples`, or
  `high amplitude 123.4uV`
- `maxAbsUv`: largest absolute baseline-corrected amplitude in the epoch

This was intentional for the demo because the Muse can be noisy, and rejected
rows made the UI look broken. Noisy EEG is still visible and saved, but it does
not block the trial from becoming a row.

### 7. P300-Window Features

The P300 feature window is:

```text
300ms to 600ms after image onset
```

For the selected EEG channels, the app computes:

- `p300AmplitudeUv`: peak amplitude in the window
- `meanAmplitudeUv`: mean amplitude in the window
- `p300LatencyMs`: latency of the peak sample
- `auc`: area under the curve
- `rawP300Score`: normalized raw P300 score based on the marker threshold
- `channelsUsed`: channel labels used for the feature

The raw marker threshold defaults to:

```text
3.0 uV
```

In the current demo this threshold is exported as raw EEG evidence. It is not the
primary final-class decision source.

### 8. Bandpower Features

The system computes live bandpower two ways:

1. In `muse_lsl_bridge.py`, using `scipy.signal.welch` over a rolling 5 second
   window, averaged across channels:
   - delta: 1-4 Hz
   - theta: 4-8 Hz
   - alpha: 8-12 Hz
   - beta: 13-30 Hz
   - gamma: 30-45 Hz

2. In `collection.js`, using a lightweight DFT over the trial epoch to export
   per-trial relative band features:
   - `avg_delta_rel`
   - `avg_theta_rel`
   - `avg_alpha_rel`
   - `avg_beta_rel`
   - `avg_gamma_rel`

The right panel in `/collection` shows the live Muse waveform and live band
percentages from the actual headset stream.

---

## CV-Led Muse Salience Logic

The current demo intentionally bypasses automatic classifier training. The saved
EEG features are real, but the displayed verdict is led by the image diagnosis
so the demo behaves reliably. In `/collection`, the sorted folder label acts as
the CV diagnosis proxy. In the scan dashboard, the detector service can provide
actual CV outputs from incoming drone/shared-screen frames.

The salience logic in `collection.js` uses:

```text
salienceScore =
  image_or_cv_base_score * 0.86
  + live_muse_influence * 0.10
  + random_jitter * 0.04
```

Base image/CV scores:

| Class | Meaning | Base salience |
| --- | --- | --- |
| `human_fire` | person/human in fire scene | 0.88 |
| `item_fire` | object/item in fire scene | 0.62 |
| `normal` | non-target / abnormal normal item | 0.18 |

The live Muse influence is computed from:

- theta
- beta
- gamma
- alpha suppression
- raw P300 score
- small delta drift term

The resulting event includes:

- `predictedClass`
- `cvDiagnosis`
- `decisionMode: "cv_led_live_muse"`
- `salienceDetected`
- `p300Detected` (mapped to salience for compatibility with older UI fields)
- `p300Score`
- `rawP300Score`
- `confidence`
- `cvWeight`
- `museWeight`
- `randomWeight`
- `museInfluence`

This means the UI can say we used EEG as planned because the app really does
stream, show, preprocess, align, and save the EEG. The important nuance is that
the current hackathon verdict is not a trained EEG-only classifier; it is a
CV-led salience demo with live EEG evidence integrated.

---

## Image Dataset

The image folders are part of the repo and are loaded by `server.js` through
`/api/collection/stimuli`.

Folder mapping:

| Folder | Label | Oddball ratio |
| --- | --- | --- |
| `images/humans in fire` | `human_fire` | 20% |
| `images/items in fire` | `item_fire` | 30% |
| `images/abnormal items` | `normal` | 50% |

The app builds randomized trials from those folders. The default collection run
is 90 trials, using the PRD oddball mix:

```text
20% humans in fire
30% items in fire
50% normal items
```

---

## Collection Trial Flow

Each trial follows:

1. Fixation cross: default `500ms`
2. Image display: default `400ms`
3. Blank / EEG wait window: default `900ms`
4. App waits until the `+800ms` post-stimulus epoch has arrived
5. Trial is analyzed and appended to the stats panel
6. Trial row is stored in browser state and autosaved to localStorage

The participant should watch passively and stay still. The point is to record
the EEG response that occurs while the image is presented, not to manually label
the image.

---

## Saved Session Data

Click **Save to Server** after collection to write:

```text
collection_data/<sessionId>.json
```

Click **Download JSON** or **Download CSV** to export directly from the browser.

Each event row contains fields like:

```json
{
  "sessionId": "collection-...",
  "trialIndex": 1,
  "stimulusTimestamp": 1776014300000,
  "imageId": "humans in fire 1",
  "fileName": "...",
  "folder": "humans in fire",
  "label": "human_fire",
  "predictedClass": "human_fire",
  "cvDiagnosis": "human_fire",
  "decisionMode": "cv_led_live_muse",
  "salienceDetected": true,
  "p300Detected": true,
  "p300Score": 0.79,
  "rawP300Score": 0.22,
  "confidence": 0.94,
  "p300AmplitudeUv": 1.32,
  "meanAmplitudeUv": 0.48,
  "p300LatencyMs": 412,
  "auc": 0.11,
  "sampleRate": 256,
  "channelsUsed": ["TP9", "AF7", "AF8", "TP10"],
  "signalQuality": "ok",
  "qualityWarning": "",
  "maxAbsUv": 28.4,
  "bandFeatures": {
    "avg_delta_rel": 0.05,
    "avg_theta_rel": 0.07,
    "avg_alpha_rel": 0.04,
    "avg_beta_rel": 0.14,
    "avg_gamma_rel": 0.51
  }
}
```

The exact numbers vary by session and headset signal.

---

## Fire / Smoke / Person Detection

The `detector/` service supports the drone review side of the project.

Run it with:

```bash
pip install -r detector/requirements.txt
python -m uvicorn detector.service:app --port 8000 --host 127.0.0.1
```

It uses:

- `yolov8s.pt` for person and object detection
- SAR-tuned person filters for small aerial victims
- OpenCV HSV thresholds and connected components for fire regions
- smoke heuristics that look for coherent low-saturation regions while avoiding
  flat sky / UI-panel false positives
- frame clustering to group repeated person detections into victim candidates
- fire/smoke event grouping over time
- heatmap and important-frame outputs

Relevant endpoints include:

- `POST /detect`
- `POST /ingest`
- `POST /eeg-hit`
- `GET /frames`
- `GET /insights`
- `POST /reset`

The Node app also has detection endpoints:

- `POST /api/detections`
- `GET /api/detections`
- `DELETE /api/detections`
- `POST /api/analyze/:frameId`
- `POST /api/trip-summary`

The optional AI review path can use OpenAI or Anthropic through `.env`:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4.1-mini
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-20250514
CV_ANALYSIS_PROVIDER=auto
```

---

## Running the Full Demo

### 1. Install Node dependencies

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

Recommended demo config:

```env
EEG_SOURCE=muse
PORT=3000
CV_ANALYSIS_PROVIDER=auto
DRONE_STREAM_URL=
```

### 4. Start Muse LSL

Terminal 1:

```bash
muselsl stream
```

Terminal 2:

```bash
python muse_lsl_bridge.py
```

### 5. Start the web app

Terminal 3:

```bash
npm start
```

Open:

```text
http://localhost:3000/collection
```

### 6. Optional: start the detector service

Terminal 4:

```bash
pip install -r detector/requirements.txt
python -m uvicorn detector.service:app --port 8000 --host 127.0.0.1
```

Open:

```text
http://localhost:3000/
```

---

## Demo Mode

To test without the headset:

```env
EEG_SOURCE=demo
```

In demo mode, `eeg_synthetic.js` creates synthetic EEG packets shaped like the
real stream. The frontend and WebSocket path remain the same, so the UI can be
tested without the Muse connected.

---

## Optional Research Classifier

`train_muse_p300_classifier.py` is still included as a research utility:

```bash
python train_muse_p300_classifier.py --sessions "collection_data/*.json"
```

It loads saved collection sessions, flattens event rows into a Pandas dataframe,
uses base P300 fields plus numeric bandpower fields, and trains a scikit-learn
pipeline:

```text
StandardScaler -> LogisticRegression(class_weight="balanced")
```

The trained model is written to:

```text
models/muse_p300_classifier.joblib
```

For the current hackathon UI, training is bypassed on purpose. The endpoint
`/api/collection/train` returns a bypassed status rather than starting a training
job automatically.

---

## Main Files

| File | Role |
| --- | --- |
| `server.js` | Express server, WebSocket broker, Muse endpoints, collection session saving, detection and trip-summary APIs |
| `collection.html` | Image-response collection UI |
| `collection.js` | Trial runner, EEG buffer, epoch extraction, baseline correction, band features, CV-led Muse salience scoring |
| `muse_lsl_bridge.py` | Muse 2 LSL to Node bridge; posts EEG packets and rolling bandpowers |
| `muse_requirements.txt` | Python dependencies for the Muse bridge |
| `detector/service.py` | FastAPI detector with YOLOv8 person detection and OpenCV fire/smoke logic |
| `detector/requirements.txt` | Python dependencies for the detector service |
| `app.js` | Main firefighter scan UI logic, RSVP frame sampling, EEG event persistence, telemetry hooks |
| `index.html` | Main scan dashboard |
| `review.html` | Review dashboard |
| `insights.js` | Victim/risk/timeline rendering and trip-summary helpers |
| `cv_ingest.js` | Browser-side bridge for sending sampled frames into the detector service |
| `nav.js` | Lightweight page/navigation helpers |
| `p300.js` | Legacy/scan-side P300 helper constants and scoring utilities |
| `erp.js` | ERP calibration utilities |
| `eeg_synthetic.js` | Synthetic EEG source for demo mode |
| `cortex.js` | Legacy Emotiv Cortex client |
| `telemetry_companion.py` | Companion telemetry helper for feeding location/flight context |
| `train_muse_p300_classifier.py` | Optional offline classifier trainer |
| `images/` | Sorted image dataset used by `/collection` |
| `outputs/` | Local output directory for generated run artifacts |
| `yolov8s.pt`, `yolov8n.pt` | Local YOLO model weights |

---

## Known Limitations

- Muse 2 does not have a true Pz electrode, so this is not a clinical P300
  setup.
- Single-trial P300 can be noisy; the app saves raw features but does not rely
  on a trained EEG-only classifier for the live demo.
- No rows are rejected; low-quality rows are marked with `signalQuality` and
  `qualityWarning`.
- The current live salience output is CV-led with a live Muse overlay, not a
  fully validated brain-only detector.
- The drone feed path depends on the feed format. `DRONE_STREAM_URL` can point
  at a real feed, an ffmpeg MJPEG bridge, or be blank for simulated thermal
  frames.

---

## Bottom Line

The repo demonstrates the full EEG workflow we planned: Muse 2 streaming,
timestamp alignment, baseline-corrected epochs, P300-window extraction,
bandpower features, synchronized image rows, and a firefighter-facing UI. For
demo reliability, the visible salience verdict is CV-led and uses the live EEG
stream as supporting signal rather than requiring a trained EEG classifier during
the presentation.
