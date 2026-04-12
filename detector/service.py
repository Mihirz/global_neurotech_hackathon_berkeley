"""
NeuroRSVP detector sidecar.

YOLOv8s for person + vehicle detection, OpenCV HSV + connected components for
fire/smoke, and a firefighter-ready insights aggregator that fuses CV
detections with EEG P300 hits and GPS telemetry.

Start:
  uvicorn detector.service:app --port 8000 --host 0.0.0.0
"""

from __future__ import annotations

import base64
import io
import time
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="NeuroRSVP detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# COCO subset relevant to firefighting SAR: people + vehicles + bags (could
# indicate a victim's belongings) + pets.
RELEVANT_COCO = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    15: "cat",
    16: "dog",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}
VICTIM_CLASSES = {"person"}

# Person-detection filters tuned for aerial / drone context.
MIN_PERSON_CONF      = 0.30
MIN_PERSON_AREA_FRAC = 0.0015   # ≥ 0.15% of frame area
MIN_PERSON_ASPECT    = 0.6      # box h/w — persons are taller than wide (even prone → 0.6+)
MAX_PERSON_ASPECT    = 5.0

_model: YOLO | None = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8s.pt")
    return _model


# ── Session state ────────────────────────────────────────────────────────────

@dataclass
class FrameRecord:
    ts: float
    frame_id: int
    width: int
    height: int
    persons: list[dict]
    other_objects: list[dict]
    fire: dict
    smoke: dict
    gps: dict | None = None
    eeg_flagged: bool = False
    eeg_amplitude: float | None = None
    eeg_score: float | None = None


@dataclass
class Session:
    started_at: float = field(default_factory=time.time)
    frames: list[FrameRecord] = field(default_factory=list)
    eeg_hits: list[dict] = field(default_factory=list)

    def reset(self):
        self.started_at = time.time()
        self.frames.clear()
        self.eeg_hits.clear()


session = Session()


# ── Image decoding ───────────────────────────────────────────────────────────

def decode_image(data_url_or_b64: str) -> np.ndarray:
    raw = data_url_or_b64
    if "," in raw and raw.startswith("data:"):
        raw = raw.split(",", 1)[1]
    img_bytes = base64.b64decode(raw)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img)


# ── Fire / smoke with HSV + connected components ─────────────────────────────

def detect_fire_smoke(rgb: np.ndarray) -> tuple[dict, dict]:
    """
    Real-flame detection:
      • HSV hue in the red→orange band (0–30) OR wrapped-red (165–179)
      • High saturation (flames are intensely colored, not pastel)
      • High value/brightness
      • Must form a connected region of reasonable size (≥ 0.3% of frame)

    Smoke detection:
      • Low saturation, mid value (grey)
      • Large contiguous coverage (≥ 10% of frame)
      • Low hue variance inside the region
      • Excluded when active fire is dominant (to avoid double-counting)
    """
    h, w = rgb.shape[:2]
    total = float(h * w)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)

    # Flame color mask: red–orange (hue 0-25) or wrapped red (170-179)
    flame_lo1 = cv2.inRange(hsv, (0,   140, 180), (25,  255, 255))
    flame_lo2 = cv2.inRange(hsv, (165, 140, 180), (179, 255, 255))
    flame_mask = cv2.bitwise_or(flame_lo1, flame_lo2)

    # Morphological cleanup to connect flame pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    flame_mask = cv2.morphologyEx(flame_mask, cv2.MORPH_OPEN, kernel)
    flame_mask = cv2.morphologyEx(flame_mask, cv2.MORPH_CLOSE, kernel)

    # Connected components — keep only sizeable flame regions
    num_f, labels_f, stats_f, _ = cv2.connectedComponentsWithStats(flame_mask, 8)
    fire_boxes: list[list[int]] = []
    fire_px = 0
    min_region = max(120, int(total * 0.0008))  # at least ~0.08% of frame per region
    for i in range(1, num_f):
        area = stats_f[i, cv2.CC_STAT_AREA]
        if area < min_region:
            continue
        x1 = int(stats_f[i, cv2.CC_STAT_LEFT])
        y1 = int(stats_f[i, cv2.CC_STAT_TOP])
        bw = int(stats_f[i, cv2.CC_STAT_WIDTH])
        bh = int(stats_f[i, cv2.CC_STAT_HEIGHT])
        fire_boxes.append([x1, y1, x1 + bw, y1 + bh])
        fire_px += int(area)

    fire_frac = fire_px / total
    fire = {
        "detected": fire_frac > 0.003 and len(fire_boxes) > 0,
        "area_frac": round(float(fire_frac), 4),
        "boxes": fire_boxes[:6],
        "intensity": round(float(V[flame_mask > 0].mean() / 255.0) if fire_px else 0.0, 3),
    }

    # Smoke: low saturation, mid-high value, coherent region
    smoke_mask = cv2.inRange(hsv, (0, 0, 70), (179, 60, 220))
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(smoke_mask, 8)
    smoke_area = 0
    smoke_boxes: list[list[int]] = []
    min_smoke = int(total * 0.03)
    for i in range(1, num_s):
        area = stats_s[i, cv2.CC_STAT_AREA]
        if area < min_smoke:
            continue
        x1 = int(stats_s[i, cv2.CC_STAT_LEFT])
        y1 = int(stats_s[i, cv2.CC_STAT_TOP])
        bw = int(stats_s[i, cv2.CC_STAT_WIDTH])
        bh = int(stats_s[i, cv2.CC_STAT_HEIGHT])
        smoke_boxes.append([x1, y1, x1 + bw, y1 + bh])
        smoke_area += int(area)

    smoke_frac = smoke_area / total
    # Guard against uniform bright skies triggering smoke — require the masked
    # region to actually have mid-range brightness variability.
    smoke_valid = smoke_frac > 0.10 and fire_frac < 0.08
    if smoke_valid:
        v_in = V[smoke_mask > 0]
        if len(v_in) > 0 and (v_in.std() < 8 or v_in.mean() > 230):
            # Flat / too-bright region — probably clear sky, not smoke.
            smoke_valid = False

    smoke = {
        "detected": smoke_valid,
        "area_frac": round(float(smoke_frac), 4),
        "boxes": smoke_boxes[:4],
    }
    return fire, smoke


# ── YOLO inference with SAR-tuned filters ────────────────────────────────────

def run_yolo(rgb: np.ndarray) -> tuple[list[dict], list[dict]]:
    model = get_model()
    h, w = rgb.shape[:2]
    total = float(h * w)
    res = model.predict(rgb, verbose=False, conf=0.25, iou=0.45, imgsz=640)[0]
    persons: list[dict] = []
    others: list[dict] = []
    if res.boxes is None:
        return persons, others

    for box in res.boxes:
        cls_id = int(box.cls.item())
        if cls_id not in RELEVANT_COCO:
            continue
        label = RELEVANT_COCO[cls_id]
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        area_frac = (bw * bh) / total
        aspect = bh / bw

        entry = {
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "conf": round(conf, 3),
            "class": label,
            "area_frac": round(area_frac, 4),
            "aspect": round(aspect, 2),
        }

        if label in VICTIM_CLASSES:
            # Reject tiny blips, UI thumbnails, and clearly non-person aspect.
            if conf < MIN_PERSON_CONF: continue
            if area_frac < MIN_PERSON_AREA_FRAC: continue
            if aspect < MIN_PERSON_ASPECT or aspect > MAX_PERSON_ASPECT: continue
            persons.append(entry)
        else:
            if conf < 0.35 or area_frac < 0.002:
                continue
            others.append(entry)

    return persons, others


# ── Request / response models ────────────────────────────────────────────────

class IngestRequest(BaseModel):
    frame_id: int
    image: str
    ts: float | None = None
    gps: dict | None = None
    eeg_flagged: bool = False
    eeg_amplitude: float | None = None
    eeg_score: float | None = None


class DetectRequest(BaseModel):
    image: str


class EEGHit(BaseModel):
    frame_id: int
    ts: float
    amplitude: float
    score: float
    gps: dict | None = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": _model is not None, "frames": len(session.frames)}


@app.post("/detect")
def detect(req: DetectRequest):
    try:
        rgb = decode_image(req.image)
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    persons, others = run_yolo(rgb)
    fire, smoke = detect_fire_smoke(rgb)
    return {
        "persons": persons,
        "other_objects": others,
        "fire": fire,
        "smoke": smoke,
        "width": rgb.shape[1],
        "height": rgb.shape[0],
    }


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        rgb = decode_image(req.image)
    except Exception as e:
        raise HTTPException(400, f"Bad image: {e}")
    persons, others = run_yolo(rgb)
    fire, smoke = detect_fire_smoke(rgb)
    record = FrameRecord(
        ts=req.ts or time.time() * 1000,
        frame_id=req.frame_id,
        width=rgb.shape[1],
        height=rgb.shape[0],
        persons=persons,
        other_objects=others,
        fire=fire,
        smoke=smoke,
        gps=req.gps,
        eeg_flagged=req.eeg_flagged,
        eeg_amplitude=req.eeg_amplitude,
        eeg_score=req.eeg_score,
    )
    session.frames.append(record)
    return {
        "ok": True,
        "frame_id": record.frame_id,
        "persons_count": len(persons),
        "vehicles_count": sum(1 for o in others if o["class"] in ("car", "truck", "bus", "motorcycle", "bicycle")),
        "fire": fire["detected"],
        "fire_area": fire["area_frac"],
        "smoke": smoke["detected"],
        "smoke_area": smoke["area_frac"],
        "session_size": len(session.frames),
    }


@app.post("/eeg-hit")
def eeg_hit(hit: EEGHit):
    session.eeg_hits.append(hit.model_dump())
    for f in session.frames[-200:]:
        if f.frame_id == hit.frame_id:
            f.eeg_flagged = True
            f.eeg_amplitude = hit.amplitude
            f.eeg_score = hit.score
            break
    return {"ok": True, "total_eeg_hits": len(session.eeg_hits)}


@app.post("/reset")
def reset():
    session.reset()
    return {"ok": True}


@app.get("/frames")
def frames(limit: int = 200):
    recent = session.frames[-limit:]
    return [
        {
            "ts": f.ts,
            "frame_id": f.frame_id,
            "persons": f.persons,
            "other_objects": f.other_objects,
            "fire": f.fire,
            "smoke": f.smoke,
            "gps": f.gps,
            "eeg_flagged": f.eeg_flagged,
            "eeg_amplitude": f.eeg_amplitude,
        }
        for f in recent
    ]


# ── Insights aggregation ─────────────────────────────────────────────────────

def _gps_key(gps: dict | None, precision: float = 2e-5) -> str | None:
    if not gps or gps.get("lat") is None or gps.get("lon") is None:
        return None
    lat = round(gps["lat"] / precision) * precision
    lon = round(gps["lon"] / precision) * precision
    return f"{lat:.6f},{lon:.6f}"


def _haversine_m(a: dict, b: dict) -> float:
    import math
    R = 6371000.0
    lat1 = math.radians(a["lat"]); lat2 = math.radians(b["lat"])
    dlat = lat2 - lat1
    dlon = math.radians(b["lon"] - a["lon"])
    s = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 2 * R * math.asin(min(1.0, s**0.5))


def _box_iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    aarea = max(1, (ax2 - ax1) * (ay2 - ay1))
    barea = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / (aarea + barea - inter + 1e-6)


def _cluster_victims(frames: list[FrameRecord]) -> list[dict]:
    items: list[dict] = []
    for f in frames:
        for p in f.persons:
            items.append({
                "ts": f.ts,
                "frame_id": f.frame_id,
                "box": p["box"],
                "conf": p["conf"],
                "area_frac": p.get("area_frac", 0),
                "gps_key": _gps_key(f.gps),
                "gps": f.gps,
                "eeg_flagged": f.eeg_flagged,
                "eeg_amplitude": f.eeg_amplitude,
            })
    if not items:
        return []

    items.sort(key=lambda it: it["ts"])
    buckets: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        buckets[it["gps_key"] or "_none"].append(it)

    clusters: list[dict] = []
    for _, bucket in buckets.items():
        tracks: list[list[dict]] = []
        for it in bucket:
            attached = False
            for track in tracks:
                last = track[-1]
                dt_ms = it["ts"] - last["ts"]
                dt = dt_ms / 1000.0 if dt_ms > 1e4 else dt_ms
                if dt < 2.5 and _box_iou(it["box"], last["box"]) > 0.2:
                    track.append(it)
                    attached = True
                    break
            if not attached:
                tracks.append([it])

        for track in tracks:
            confs = [t["conf"] for t in track]
            eeg_support = sum(1 for t in track if t["eeg_flagged"])
            max_conf = max(confs)
            avg_conf = sum(confs) / len(confs)
            priority = min(1.0, (0.4 + 0.6 * max_conf) * (1 + 0.4 * min(eeg_support, 3) / 3))
            avg_area = sum(t["area_frac"] for t in track) / len(track)
            first = track[0]; last = track[-1]
            clusters.append({
                "victim_id": f"V{len(clusters)+1:03d}",
                "detections": len(track),
                "first_seen_ts": first["ts"],
                "last_seen_ts": last["ts"],
                "first_frame": first["frame_id"],
                "last_frame": last["frame_id"],
                "max_confidence": round(max_conf, 3),
                "avg_confidence": round(avg_conf, 3),
                "avg_area_frac": round(avg_area, 4),
                "eeg_corroborations": eeg_support,
                "gps": first["gps"],
                "priority": round(priority, 3),
            })

    clusters.sort(key=lambda c: (-c["priority"], -c["detections"]))
    return clusters


def _fire_events(frames: list[FrameRecord]) -> list[dict]:
    events: list[dict] = []
    current: dict | None = None
    for f in frames:
        hot = f.fire["detected"] or f.smoke["detected"]
        if hot:
            kind = "fire" if f.fire["detected"] else "smoke"
            severity = f.fire["area_frac"] if f.fire["detected"] else f.smoke["area_frac"]
            gap = (f.ts - current["last_ts"]) if current else 0
            if current and current["kind"] == kind and gap < 3000:
                current["last_ts"] = f.ts
                current["last_frame"] = f.frame_id
                current["max_severity"] = max(current["max_severity"], severity)
                current["frame_count"] += 1
                if f.gps and not current.get("gps"):
                    current["gps"] = f.gps
            else:
                if current:
                    events.append(current)
                current = {
                    "kind": kind,
                    "first_ts": f.ts,
                    "last_ts": f.ts,
                    "first_frame": f.frame_id,
                    "last_frame": f.frame_id,
                    "max_severity": severity,
                    "frame_count": 1,
                    "gps": f.gps,
                }
    if current:
        events.append(current)
    for ev in events:
        ev["max_severity"] = round(ev["max_severity"], 4)
        # Duration estimate from frame count (ingest runs ~2 Hz)
        ev["duration_s"] = round((ev["last_ts"] - ev["first_ts"]) / 1000.0
                                  if ev["last_ts"] > 1e10 else ev["last_ts"] - ev["first_ts"], 1)
    return events


def _timeline(victims, risks, eeg_hits):
    events = []
    for v in victims:
        events.append({
            "t": v["first_seen_ts"],
            "type": "victim_detected",
            "label": f"{v['victim_id']} detected (conf {v['max_confidence']:.2f}, {v['detections']} hits)",
            "priority": v["priority"],
            "gps": v["gps"],
        })
    for r in risks:
        events.append({
            "t": r["first_ts"],
            "type": f"risk_{r['kind']}",
            "label": f"{r['kind'].upper()} — severity {r['max_severity']*100:.1f}% across {r['frame_count']} frames",
            "priority": min(1.0, r["max_severity"] * 12),
            "gps": r.get("gps"),
        })
    for h in eeg_hits:
        events.append({
            "t": h["ts"],
            "type": "eeg_p300",
            "label": f"Firefighter P300 hit — {h['amplitude']:.1f}µV",
            "priority": min(1.0, abs(h["amplitude"]) / 10),
            "gps": h.get("gps"),
        })
    events.sort(key=lambda e: e["t"])
    return events


def _heatmap(frames):
    counts: dict[str, dict] = {}
    for f in frames:
        key = _gps_key(f.gps)
        if not key:
            continue
        c = counts.setdefault(key, {
            "gps": f.gps, "persons": 0, "fire": 0, "smoke": 0, "frames": 0,
        })
        c["frames"] += 1
        c["persons"] += len(f.persons)
        c["fire"]   += int(f.fire["detected"])
        c["smoke"]  += int(f.smoke["detected"])
    return list(counts.values())


def _flight_stats(frames: list[FrameRecord], victims: list[dict], risks: list[dict]):
    if not frames:
        return {}
    total = len(frames)
    person_frames   = sum(1 for f in frames if f.persons)
    fire_frames     = sum(1 for f in frames if f.fire["detected"])
    smoke_frames    = sum(1 for f in frames if f.smoke["detected"])
    vehicle_frames  = sum(1 for f in frames if any(o["class"] in ("car","truck","bus","motorcycle","bicycle") for o in f.other_objects))

    all_persons = [p for f in frames for p in f.persons]
    avg_conf = (sum(p["conf"] for p in all_persons) / len(all_persons)) if all_persons else 0.0
    peak = max((len(f.persons) for f in frames), default=0)
    avg_per_frame = len(all_persons) / total

    ts_first = frames[0].ts
    ts_last  = frames[-1].ts
    duration_s = (ts_last - ts_first) / 1000.0 if ts_last > 1e10 else (ts_last - ts_first)
    duration_s = max(0.1, duration_s)

    # Coverage area via GPS bbox
    gpss = [f.gps for f in frames if f.gps and f.gps.get("lat") is not None]
    coverage_m = 0.0
    bbox = None
    if len(gpss) >= 2:
        lats = [g["lat"] for g in gpss]; lons = [g["lon"] for g in gpss]
        bbox = {"min_lat": min(lats), "max_lat": max(lats),
                "min_lon": min(lons), "max_lon": max(lons)}
        coverage_m = _haversine_m({"lat": bbox["min_lat"], "lon": bbox["min_lon"]},
                                   {"lat": bbox["max_lat"], "lon": bbox["max_lon"]})

    time_since_last_victim = None
    if victims:
        last_ts = max(v["last_seen_ts"] for v in victims)
        now_ts = frames[-1].ts
        dt = (now_ts - last_ts) / 1000.0 if now_ts > 1e10 else (now_ts - last_ts)
        time_since_last_victim = round(max(0.0, dt), 1)

    critical = sum(1 for v in victims if v["priority"] >= 0.8)
    high     = sum(1 for v in victims if 0.5 <= v["priority"] < 0.8)

    return {
        "total_frames": total,
        "duration_s": round(duration_s, 1),
        "person_frames": person_frames,
        "vehicle_frames": vehicle_frames,
        "fire_frames": fire_frames,
        "smoke_frames": smoke_frames,
        "victims_count": len(victims),
        "critical_victims": critical,
        "high_priority_victims": high,
        "eeg_corroborated_victims": sum(1 for v in victims if v["eeg_corroborations"] > 0),
        "fire_events": len([r for r in risks if r["kind"] == "fire"]),
        "smoke_events": len([r for r in risks if r["kind"] == "smoke"]),
        "avg_confidence": round(avg_conf, 3),
        "peak_simultaneous_persons": peak,
        "avg_persons_per_frame": round(avg_per_frame, 2),
        "person_detection_rate": round(person_frames / total, 3),
        "fire_coverage_pct": round(100 * fire_frames / total, 1),
        "smoke_coverage_pct": round(100 * smoke_frames / total, 1),
        "detections_per_min": round(len(all_persons) / (duration_s / 60.0), 1) if duration_s > 0 else 0,
        "gps_bbox": bbox,
        "coverage_diagonal_m": round(coverage_m, 1),
        "time_since_last_victim_s": time_since_last_victim,
    }


@app.get("/insights")
def insights():
    frames = session.frames
    victims = _cluster_victims(frames)
    risks = _fire_events(frames)
    timeline = _timeline(victims, risks, session.eeg_hits)
    heatmap = _heatmap(frames)
    stats = _flight_stats(frames, victims, risks)

    lines: list[str] = []
    if frames:
        lines.append(
            f"{stats['victims_count']} victim(s) across {stats['total_frames']} frames in "
            f"{stats['duration_s']:.0f}s"
            + (f" — {stats['critical_victims']} CRITICAL, {stats['high_priority_victims']} HIGH." if stats['victims_count'] else ".")
        )
        if stats["eeg_corroborated_victims"]:
            lines.append(f"{stats['eeg_corroborated_victims']} corroborated by firefighter P300.")
        if stats["fire_events"] or stats["smoke_events"]:
            lines.append(
                f"{stats['fire_events']} fire event(s) ({stats['fire_coverage_pct']}% of flight), "
                f"{stats['smoke_events']} smoke event(s) ({stats['smoke_coverage_pct']}% of flight)."
            )
        if stats["peak_simultaneous_persons"]:
            lines.append(
                f"Peak {stats['peak_simultaneous_persons']} simultaneous persons, "
                f"avg confidence {stats['avg_confidence']*100:.0f}%."
            )
        if stats["coverage_diagonal_m"]:
            lines.append(f"Coverage area: {stats['coverage_diagonal_m']:.0f} m diagonal.")
        if stats["time_since_last_victim_s"] is not None:
            lines.append(f"Last victim seen {stats['time_since_last_victim_s']:.0f}s ago.")
    else:
        lines.append("Awaiting drone feed…")

    summary = {**stats, "lines": lines}

    return {
        "summary": summary,
        "victims": victims,
        "risks": risks,
        "timeline": timeline,
        "heatmap": heatmap,
    }
