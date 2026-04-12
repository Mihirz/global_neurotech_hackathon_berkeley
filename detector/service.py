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
from pathlib import Path

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
MIN_PERSON_CONF      = 0.24
MIN_PERSON_AREA_FRAC = 0.00025  # allow smaller aerial victims; clusters handle false positives
MIN_PERSON_ASPECT    = 0.35     # prone/crouched victims can be wider than upright people
MAX_PERSON_ASPECT    = 6.5
MIN_VICTIM_TRACK_HITS = 2

_model: YOLO | None = None
MODEL_PATH = Path(__file__).resolve().parent.parent / "yolov8s.pt"


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
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
    thumbnail: str | None = None


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


def encode_thumbnail(rgb: np.ndarray, max_width: int = 360) -> str:
    img = Image.fromarray(rgb)
    if img.width > max_width:
        scale = max_width / float(img.width)
        img = img.resize((max_width, max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=58, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ── Fire / smoke with HSV + connected components ─────────────────────────────

def _component_touch_count(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> int:
    pad = max(3, int(min(img_w, img_h) * 0.01))
    return int(x <= pad) + int(y <= pad) + int(x + w >= img_w - pad) + int(y + h >= img_h - pad)


def _component_texture(gray: np.ndarray, mask: np.ndarray) -> float:
    vals = gray[mask]
    return float(vals.std()) if vals.size else 0.0

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
      • Tracked independently from active fire when the smoke evidence is strong
    """
    h, w = rgb.shape[:2]
    total = float(h * w)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
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
    fire_scores: list[float] = []
    min_region = max(120, int(total * 0.0008))  # at least ~0.08% of frame per region
    for i in range(1, num_f):
        area = stats_f[i, cv2.CC_STAT_AREA]
        if area < min_region:
            continue
        x1 = int(stats_f[i, cv2.CC_STAT_LEFT])
        y1 = int(stats_f[i, cv2.CC_STAT_TOP])
        bw = int(stats_f[i, cv2.CC_STAT_WIDTH])
        bh = int(stats_f[i, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, bw * bh)
        extent = float(area / bbox_area)
        aspect = bw / max(1, bh)
        comp_mask = labels_f == i
        hue_std = float(H[comp_mask].std()) if area else 0.0
        val_std = float(V[comp_mask].std()) if area else 0.0
        texture = _component_texture(gray, comp_mask)
        touches = _component_touch_count(x1, y1, bw, bh, w, h)

        # Bright app banners, warning labels, and UI accents tend to be solid,
        # rectangular, and low-texture. Flame blobs are irregular and textured.
        ui_like_rect = extent > 0.78 and (aspect > 2.4 or aspect < 0.42) and texture < 18
        flat_color = hue_std < 3.0 and val_std < 10.0 and texture < 12.0
        edge_badge = touches >= 2 and extent > 0.72 and texture < 18
        if ui_like_rect or flat_color or edge_badge:
            continue

        fire_boxes.append([x1, y1, x1 + bw, y1 + bh])
        fire_px += int(area)
        fire_scores.append(min(1.0, (area / total) * 55 + texture / 95))

    fire_frac = fire_px / total
    fire_conf = max(fire_scores, default=0.0)
    fire = {
        "detected": fire_frac > 0.0025 and fire_conf >= 0.20 and len(fire_boxes) > 0,
        "area_frac": round(float(fire_frac), 4),
        "boxes": fire_boxes[:6],
        "intensity": round(float(V[flame_mask > 0].mean() / 255.0) if fire_px else 0.0, 3),
        "confidence": round(float(fire_conf), 3),
    }

    # Smoke: low saturation, mid-high value, coherent region
    smoke_mask = cv2.inRange(hsv, (0, 0, 70), (179, 60, 220))
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    num_s, labels_s, stats_s, _ = cv2.connectedComponentsWithStats(smoke_mask, 8)
    smoke_area = 0
    smoke_boxes: list[list[int]] = []
    smoke_scores: list[float] = []
    min_smoke = int(total * 0.03)
    for i in range(1, num_s):
        area = stats_s[i, cv2.CC_STAT_AREA]
        if area < min_smoke:
            continue
        x1 = int(stats_s[i, cv2.CC_STAT_LEFT])
        y1 = int(stats_s[i, cv2.CC_STAT_TOP])
        bw = int(stats_s[i, cv2.CC_STAT_WIDTH])
        bh = int(stats_s[i, cv2.CC_STAT_HEIGHT])
        bbox_area = max(1, bw * bh)
        extent = float(area / bbox_area)
        comp_mask = labels_s == i
        gray_std = _component_texture(gray, comp_mask)
        val_mean = float(V[comp_mask].mean()) if area else 0.0
        touches = _component_touch_count(x1, y1, bw, bh, w, h)

        # Reject flat grey UI panels / clear sky. Smoke is diffuse, but it still
        # has rolling brightness variation and rarely fills a perfect rectangle.
        if gray_std < 10.0:
            continue
        if extent > 0.88 and gray_std < 18.0:
            continue
        if touches >= 3 and area / total > 0.45:
            continue
        if val_mean > 225:
            continue

        smoke_boxes.append([x1, y1, x1 + bw, y1 + bh])
        smoke_area += int(area)
        smoke_scores.append(min(1.0, (area / total) * 4 + gray_std / 80))

    smoke_frac = smoke_area / total
    smoke_conf = max(smoke_scores, default=0.0)
    # Guard against uniform bright skies triggering smoke — require the masked
    # region to actually have mid-range brightness variability.
    smoke_valid = smoke_frac > 0.12 and smoke_conf >= 0.22
    if smoke_valid:
        v_in = V[smoke_mask > 0]
        if len(v_in) > 0 and (v_in.std() < 8 or v_in.mean() > 230):
            # Flat / too-bright region — probably clear sky, not smoke.
            smoke_valid = False
        if fire_frac >= 0.08 and smoke_frac < 0.20 and smoke_conf < 0.45:
            # Flames can turn nearby shadows grey; require stronger evidence
            # before also calling smoke in heavily fire-dominant closeups.
            smoke_valid = False

    smoke = {
        "detected": smoke_valid,
        "area_frac": round(float(smoke_frac), 4),
        "boxes": smoke_boxes[:4],
        "confidence": round(float(smoke_conf), 3),
    }
    return fire, smoke


# ── YOLO inference with SAR-tuned filters ────────────────────────────────────

def run_yolo(rgb: np.ndarray) -> tuple[list[dict], list[dict]]:
    model = get_model()
    h, w = rgb.shape[:2]
    total = float(h * w)
    res = model.predict(
        rgb,
        verbose=False,
        conf=0.18,
        iou=0.45,
        imgsz=960,
        classes=list(RELEVANT_COCO.keys()),
    )[0]
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
            # Final rescue-target confirmation happens across frames in
            # _cluster_victims so distant aerial people can still pass here.
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
        thumbnail=encode_thumbnail(rgb),
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
    hit_data = hit.model_dump() if hasattr(hit, "model_dump") else hit.dict()
    session.eeg_hits.append(hit_data)
    matched = False
    for f in session.frames[-200:]:
        if f.frame_id == hit.frame_id:
            f.eeg_flagged = True
            f.eeg_amplitude = hit.amplitude
            f.eeg_score = hit.score
            matched = True
            break
    if not matched and session.frames:
        nearest = min(session.frames[-200:], key=lambda f: abs(f.ts - hit.ts))
        if abs(nearest.ts - hit.ts) <= 750:
            nearest.eeg_flagged = True
            nearest.eeg_amplitude = hit.amplitude
            nearest.eeg_score = hit.score
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
            if len(track) < MIN_VICTIM_TRACK_HITS and max_conf < 0.68 and eeg_support == 0:
                continue
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
    active: dict[str, dict | None] = {"fire": None, "smoke": None}

    def flush(kind: str):
        current = active[kind]
        if not current:
            return
        # One-frame UI flashes should not become dispatch-level events unless
        # the region is large enough to be operationally obvious.
        if current["frame_count"] >= 2 or current["max_severity"] >= 0.08:
            events.append(current.copy())
        active[kind] = None

    for f in frames:
        for kind, detected, severity in (
            ("fire", f.fire["detected"], f.fire["area_frac"]),
            ("smoke", f.smoke["detected"], f.smoke["area_frac"]),
        ):
            current = active[kind]
            if detected:
                gap = (f.ts - current["last_ts"]) if current else 0
                if current and gap < 3000:
                    current["last_ts"] = f.ts
                    current["last_frame"] = f.frame_id
                    current["max_severity"] = max(current["max_severity"], severity)
                    current["frame_count"] += 1
                    if f.gps and not current.get("gps"):
                        current["gps"] = f.gps
                else:
                    flush(kind)
                    active[kind] = {
                        "kind": kind,
                        "first_ts": f.ts,
                        "last_ts": f.ts,
                        "first_frame": f.frame_id,
                        "last_frame": f.frame_id,
                        "max_severity": severity,
                        "frame_count": 1,
                        "gps": f.gps,
                    }
            else:
                flush(kind)
    flush("fire")
    flush("smoke")
    for ev in events:
        ev["max_severity"] = round(ev["max_severity"], 4)
        # Duration estimate from frame count (ingest runs ~2 Hz)
        ev["duration_s"] = round((ev["last_ts"] - ev["first_ts"]) / 1000.0
                                  if ev["last_ts"] > 1e10 else ev["last_ts"] - ev["first_ts"], 1)
    events.sort(key=lambda ev: ev["first_ts"])
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


def _important_frames(frames: list[FrameRecord], eeg_hits: list[dict]) -> list[dict]:
    items: list[dict] = []
    eeg_by_frame = {h.get("frame_id"): h for h in eeg_hits}

    for f in frames:
        reasons: list[str] = []
        score = 0.0
        details: dict = {}

        if f.persons:
            max_conf = max(p["conf"] for p in f.persons)
            reasons.append(f"{len(f.persons)} person candidate{'s' if len(f.persons) != 1 else ''}")
            score = max(score, 0.45 + max_conf * 0.45)
            details["person_count"] = len(f.persons)
            details["person_confidence"] = round(max_conf, 3)

        if f.fire["detected"]:
            severity = f.fire["area_frac"]
            reasons.append(f"fire {severity * 100:.1f}%")
            score = max(score, min(1.0, 0.35 + severity * 8))
            details["fire_area_frac"] = severity
            details["fire_confidence"] = f.fire.get("confidence")

        if f.smoke["detected"]:
            severity = f.smoke["area_frac"]
            reasons.append(f"smoke {severity * 100:.1f}%")
            score = max(score, min(1.0, 0.30 + severity * 3))
            details["smoke_area_frac"] = severity
            details["smoke_confidence"] = f.smoke.get("confidence")

        hit = eeg_by_frame.get(f.frame_id)
        if f.eeg_flagged or hit:
            amp = f.eeg_amplitude if f.eeg_amplitude is not None else hit.get("amplitude") if hit else None
            reasons.append(f"P300 {amp:.1f}uV" if amp is not None else "P300")
            score = max(score, 0.80)
            details["eeg_amplitude"] = amp

        if not reasons:
            continue

        items.append({
            "frame_id": f.frame_id,
            "ts": f.ts,
            "reasons": reasons,
            "score": round(score, 3),
            "gps": f.gps,
            "details": details,
            "thumbnail": f.thumbnail,
        })

    items.sort(key=lambda it: (-it["score"], it["frame_id"]))
    return items[:24]


def _flight_stats(frames: list[FrameRecord], victims: list[dict], risks: list[dict]):
    if not frames:
        return {}
    total = len(frames)
    person_frames   = sum(1 for f in frames if f.persons)
    fire_frames     = sum(1 for f in frames if f.fire["detected"])
    smoke_frames    = sum(1 for f in frames if f.smoke["detected"])
    vehicle_frames  = sum(1 for f in frames if any(o["class"] in ("car","truck","bus","motorcycle","bicycle") for o in f.other_objects))
    risk_frames     = sum(1 for f in frames if f.fire["detected"] or f.smoke["detected"])
    clear_frames    = total - risk_frames
    max_fire_sev    = max((f.fire["area_frac"] for f in frames if f.fire["detected"]), default=0.0)
    max_smoke_sev   = max((f.smoke["area_frac"] for f in frames if f.smoke["detected"]), default=0.0)

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
    top_risk = max([v["priority"] for v in victims] + [min(1.0, r["max_severity"] * 12) for r in risks] + [0.0])
    risk_score = round(min(100.0, top_risk * 65 + critical * 20 + high * 10 + len(risks) * 6), 1)
    fps_observed = total / duration_s if duration_s > 0 else 0.0
    fire_exposure_s = round(duration_s * fire_frames / total, 1)
    smoke_exposure_s = round(duration_s * smoke_frames / total, 1)

    return {
        "total_frames": total,
        "duration_s": round(duration_s, 1),
        "person_frames": person_frames,
        "vehicle_frames": vehicle_frames,
        "fire_frames": fire_frames,
        "smoke_frames": smoke_frames,
        "risk_frames": risk_frames,
        "clear_frames": clear_frames,
        "victims_count": len(victims),
        "critical_victims": critical,
        "high_priority_victims": high,
        "eeg_corroborated_victims": sum(1 for v in victims if v["eeg_corroborations"] > 0),
        "fire_events": len([r for r in risks if r["kind"] == "fire"]),
        "smoke_events": len([r for r in risks if r["kind"] == "smoke"]),
        "max_fire_severity": round(max_fire_sev, 4),
        "max_smoke_severity": round(max_smoke_sev, 4),
        "fire_exposure_s": fire_exposure_s,
        "smoke_exposure_s": smoke_exposure_s,
        "risk_score": risk_score,
        "avg_confidence": round(avg_conf, 3),
        "peak_simultaneous_persons": peak,
        "avg_persons_per_frame": round(avg_per_frame, 2),
        "person_detection_rate": round(person_frames / total, 3),
        "fire_coverage_pct": round(100 * fire_frames / total, 1),
        "smoke_coverage_pct": round(100 * smoke_frames / total, 1),
        "detections_per_min": round(len(all_persons) / (duration_s / 60.0), 1) if duration_s > 0 else 0,
        "observed_fps": round(fps_observed, 2),
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
    important_frames = _important_frames(frames, session.eeg_hits)
    stats = _flight_stats(frames, victims, risks)

    lines: list[str] = []
    if frames:
        lines.append(
            f"{stats['victims_count']} victim(s) across {stats['total_frames']} frames in "
            f"{stats['duration_s']:.0f}s"
            + (f" — {stats['critical_victims']} CRITICAL, {stats['high_priority_victims']} HIGH." if stats['victims_count'] else ".")
        )
        lines.append(
            f"Scan quality: {stats['observed_fps']:.1f} fps observed, "
            f"{stats['person_frames']} person-candidate frame(s), risk score {stats['risk_score']:.0f}/100."
        )
        if stats["eeg_corroborated_victims"]:
            lines.append(f"{stats['eeg_corroborated_victims']} corroborated by firefighter P300.")
        if stats["fire_events"] or stats["smoke_events"]:
            lines.append(
                f"{stats['fire_events']} fire event(s) ({stats['fire_coverage_pct']}% of flight), "
                f"{stats['smoke_events']} smoke event(s) ({stats['smoke_coverage_pct']}% of flight); "
                f"peak fire {stats['max_fire_severity']*100:.1f}%, peak smoke {stats['max_smoke_severity']*100:.1f}%."
            )
        elif stats["person_frames"] == 0:
            lines.append("No persistent person or hazard tracks yet; continue a slow orbit and keep the camera clear of app UI overlays.")
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
        "important_frames": important_frames,
    }
