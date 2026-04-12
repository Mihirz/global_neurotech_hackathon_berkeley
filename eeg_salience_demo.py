"""
EEG salience demo for fire-scene image classification.

This is a mock cognitive detection system for the hackathon prototype. It uses
the sorted image folders as stimulus classes, simulates P300-shaped EEG by
default, trains per-user demo classifiers, and emits real-time JSONL event flags
aligned to image timestamps.

Run:
  python eeg_salience_demo.py --dry-run --trials 18
  python eeg_salience_demo.py --display --trials 60

The script intentionally keeps LSL / real headset integration optional. For a
real stream, replace SyntheticEEGSource with an LSL-backed source that exposes
the same get_epoch(label, pre_s, post_s) method.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SAMPLE_RATE_HZ = 256
CHANNELS = ("Fz", "Cz", "Pz")
PZ_INDEX = CHANNELS.index("Pz")

PRE_STIM_S = 0.2
POST_STIM_S = 0.8
P300_WIN_S = (0.300, 0.600)

CATEGORY_CONFIG = {
    # Folder name -> class metadata.
    # The existing repo calls the non-target folder "abnormal items"; for this
    # PRD pipeline it is treated as the frequent / lowest-salience baseline.
    "humans in fire": {
        "label": "human_fire",
        "oddball_weight": 0.20,
        "sim_p300_uv": 9.0,
        "target_rank": 2,
    },
    "items in fire": {
        "label": "item_fire",
        "oddball_weight": 0.30,
        "sim_p300_uv": 5.0,
        "target_rank": 1,
    },
    "abnormal items": {
        "label": "normal",
        "oddball_weight": 0.50,
        "sim_p300_uv": 1.2,
        "target_rank": 0,
    },
}


@dataclass(frozen=True)
class Stimulus:
    image_id: str
    path: Path
    folder: str
    label: str
    target_rank: int
    sim_p300_uv: float


@dataclass
class Event:
    timestamp: float
    image_id: str
    true_class: str
    predicted_class: str
    confidence: float
    p300_detected: bool
    p300_strength: float
    p300_peak_latency_ms: float
    mean_amplitude_uv: float
    peak_amplitude_uv: float

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True)


def is_image_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_image_dataset(root: Path) -> dict[str, list[Stimulus]]:
    images_root = root / "images"
    dataset: dict[str, list[Stimulus]] = {}
    for folder, cfg in CATEGORY_CONFIG.items():
        folder_path = images_root / folder
        if not folder_path.exists():
            raise FileNotFoundError(f"Missing image category folder: {folder_path}")
        stimuli = []
        for path in sorted(folder_path.iterdir(), key=lambda p: p.name.lower()):
            if is_image_file(path):
                stimuli.append(
                    Stimulus(
                        image_id=path.stem,
                        path=path,
                        folder=folder,
                        label=cfg["label"],
                        target_rank=cfg["target_rank"],
                        sim_p300_uv=cfg["sim_p300_uv"],
                    )
                )
        if not stimuli:
            raise ValueError(f"No image stimuli found in {folder_path}")
        dataset[cfg["label"]] = stimuli
    return dataset


def make_oddball_trials(
    dataset: dict[str, list[Stimulus]],
    n_trials: int,
    rng: random.Random,
) -> list[Stimulus]:
    labels = [cfg["label"] for cfg in CATEGORY_CONFIG.values()]
    weights = [cfg["oddball_weight"] for cfg in CATEGORY_CONFIG.values()]
    counts = [int(round(n_trials * weight)) for weight in weights]
    if n_trials >= len(labels):
        counts = [max(1, count) for count in counts]
    while sum(counts) > n_trials:
        idx = max(range(len(counts)), key=lambda i: counts[i] if counts[i] > 1 else -1)
        counts[idx] -= 1
    while sum(counts) < n_trials:
        counts[weights.index(max(weights))] += 1

    trials = []
    for label, count in zip(labels, counts):
        for _ in range(count):
            trials.append(rng.choice(dataset[label]))
    # Prevent long runs of one class; P300 demo works best with scattered oddballs.
    rng.shuffle(trials)
    return trials


class SyntheticEEGSource:
    """Generate baseline + class-dependent P300 epochs at 256 Hz."""

    def __init__(self, sample_rate_hz: int = SAMPLE_RATE_HZ, seed: int = 7):
        self.sample_rate_hz = sample_rate_hz
        self.rng = np.random.default_rng(seed)
        self.times = np.arange(
            -PRE_STIM_S,
            POST_STIM_S,
            1.0 / self.sample_rate_hz,
            dtype=np.float64,
        )

    def get_epoch(self, stimulus: Stimulus, pre_s: float = PRE_STIM_S, post_s: float = POST_STIM_S) -> np.ndarray:
        del pre_s, post_s
        noise = self.rng.normal(0, 1.8, size=(len(CHANNELS), len(self.times)))
        slow = self.rng.normal(0, 0.6, size=(len(CHANNELS), 1)) * np.sin(2 * math.pi * 1.2 * self.times)

        latency = self.rng.normal(0.410, 0.035)
        width = self.rng.uniform(0.055, 0.080)
        p300 = stimulus.sim_p300_uv * np.exp(-((self.times - latency) ** 2) / (2 * width * width))

        # P300 is strongest at Pz, visible at Cz, weaker at Fz.
        weights = np.array([0.30, 0.65, 1.00]).reshape(-1, 1)
        epoch = noise + slow + weights * p300

        # Occasional artifact, mostly frontal, to keep artifact rejection real.
        if self.rng.random() < 0.03:
            epoch[0] += self.rng.normal(55, 12) * np.exp(-((self.times - 0.12) ** 2) / (2 * 0.025 * 0.025))
        return epoch


def bandpass_filter(eeg: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ) -> np.ndarray:
    nyq = sample_rate_hz / 2.0
    b, a = butter(3, [0.1 / nyq, 30.0 / nyq], btype="band")
    return filtfilt(b, a, eeg, axis=1)


def preprocess(eeg: np.ndarray, times: np.ndarray) -> np.ndarray | None:
    if np.max(np.abs(eeg)) > 100.0:
        return None
    filtered = bandpass_filter(eeg)
    baseline_mask = times < 0
    baseline = filtered[:, baseline_mask].mean(axis=1, keepdims=True)
    corrected = filtered - baseline
    return corrected


def extract_features(eeg: np.ndarray, times: np.ndarray) -> np.ndarray:
    win_mask = (times >= P300_WIN_S[0]) & (times <= P300_WIN_S[1])
    window = eeg[:, win_mask]
    pz = eeg[PZ_INDEX, win_mask]
    pz_times = times[win_mask]

    pz_peak_idx = int(np.argmax(pz))
    pz_peak = float(pz[pz_peak_idx])
    pz_latency_ms = float(pz_times[pz_peak_idx] * 1000.0)
    pz_mean = float(pz.mean())
    if hasattr(np, "trapezoid"):
        pz_auc = float(np.trapezoid(pz, pz_times))
    else:
        pz_auc = float(np.trapz(pz, pz_times))

    channel_means = window.mean(axis=1)
    channel_peaks = window.max(axis=1)
    return np.array([pz_mean, pz_peak, pz_latency_ms, pz_auc, *channel_means, *channel_peaks], dtype=np.float64)


def feature_summary(features: np.ndarray) -> dict[str, float]:
    return {
        "mean_amplitude_uv": round(float(features[0]), 3),
        "peak_amplitude_uv": round(float(features[1]), 3),
        "peak_latency_ms": round(float(features[2]), 1),
        "auc": round(float(features[3]), 4),
    }


def collect_training_set(
    source: SyntheticEEGSource,
    trials: Iterable[Stimulus],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Stimulus]]:
    xs: list[np.ndarray] = []
    ys_class: list[str] = []
    ys_p300: list[int] = []
    kept: list[Stimulus] = []
    for stim in trials:
        eeg = source.get_epoch(stim)
        processed = preprocess(eeg, source.times)
        if processed is None:
            continue
        xs.append(extract_features(processed, source.times))
        ys_class.append(stim.label)
        ys_p300.append(1 if stim.target_rank > 0 else 0)
        kept.append(stim)
    if not xs:
        feature_count = 4 + (2 * len(CHANNELS))
        return np.empty((0, feature_count)), np.array([]), np.array([]), kept
    return np.vstack(xs), np.array(ys_class), np.array(ys_p300), kept


def train_models(x: np.ndarray, y_class: np.ndarray, y_p300: np.ndarray):
    expected_classes = {cfg["label"] for cfg in CATEGORY_CONFIG.values()}
    if set(y_class) != expected_classes:
        missing = sorted(expected_classes - set(y_class))
        raise RuntimeError(f"Calibration data missing class samples: {missing}")
    if len(set(y_p300)) < 2:
        raise RuntimeError("Calibration data needs both P300 and non-P300 examples")

    p300_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    class_model = make_pipeline(
        StandardScaler(),
        LinearDiscriminantAnalysis(),
    )
    p300_model.fit(x, y_p300)
    class_model.fit(x, y_class)
    return p300_model, class_model


def predict_event(
    stimulus: Stimulus,
    timestamp: float,
    source: SyntheticEEGSource,
    p300_model,
    class_model,
) -> Event | None:
    eeg = source.get_epoch(stimulus)
    processed = preprocess(eeg, source.times)
    if processed is None:
        return None
    features = extract_features(processed, source.times)
    p300_prob = float(p300_model.predict_proba(features.reshape(1, -1))[0][1])
    class_probs = class_model.predict_proba(features.reshape(1, -1))[0]
    class_labels = class_model.classes_
    best_idx = int(np.argmax(class_probs))
    predicted = str(class_labels[best_idx])
    confidence = float(class_probs[best_idx])
    fs = feature_summary(features)
    return Event(
        timestamp=round(timestamp, 4),
        image_id=stimulus.image_id,
        true_class=stimulus.label,
        predicted_class=predicted,
        confidence=round(confidence, 3),
        p300_detected=bool(p300_prob >= 0.55),
        p300_strength=round(p300_prob, 3),
        p300_peak_latency_ms=fs["peak_latency_ms"],
        mean_amplitude_uv=fs["mean_amplitude_uv"],
        peak_amplitude_uv=fs["peak_amplitude_uv"],
    )


def maybe_show_image(stimulus: Stimulus, display: bool, fixation_s: float, image_s: float, blank_s: float) -> None:
    if not display:
        return
    try:
        import pygame
    except ImportError as exc:
        raise RuntimeError("Install pygame or run without --display") from exc

    if not pygame.get_init():
        pygame.init()
        pygame.display.set_caption("EEG Salience Stimulus")
        maybe_show_image.screen = pygame.display.set_mode((960, 720))
    screen = getattr(maybe_show_image, "screen", None)
    if screen is None:
        screen = pygame.display.get_surface() or pygame.display.set_mode((960, 720))
        maybe_show_image.screen = screen
    screen.fill((0, 0, 0))
    font = pygame.font.SysFont("Arial", 64)
    cross = font.render("+", True, (230, 230, 230))
    screen.blit(cross, cross.get_rect(center=screen.get_rect().center))
    pygame.display.flip()
    time.sleep(fixation_s)

    with Image.open(stimulus.path) as pil_image:
        pil_image = pil_image.convert("RGB")
        image = pygame.image.fromstring(pil_image.tobytes(), pil_image.size, "RGB")
    image_rect = image.get_rect()
    scale = min(screen.get_width() / image_rect.width, screen.get_height() / image_rect.height)
    size = (int(image_rect.width * scale), int(image_rect.height * scale))
    image = pygame.transform.smoothscale(image, size)
    screen.fill((0, 0, 0))
    screen.blit(image, image.get_rect(center=screen.get_rect().center))
    pygame.display.flip()
    time.sleep(image_s)
    screen.fill((0, 0, 0))
    pygame.display.flip()
    time.sleep(blank_s)


def run_session(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    root = Path(args.repo_root).resolve()
    dataset = load_image_dataset(root)
    source = SyntheticEEGSource(seed=args.seed)

    train_trials = make_oddball_trials(dataset, args.calibration_trials, rng)
    x_train, y_class, y_p300, kept = collect_training_set(source, train_trials)
    if len(kept) < 20:
        raise RuntimeError("Too few valid calibration epochs after artifact rejection")
    p300_model, class_model = train_models(x_train, y_class, y_p300)

    print(
        json.dumps(
            {
                "type": "calibration_complete",
                "valid_epochs": len(kept),
                "class_counts": {label: int((y_class == label).sum()) for label in sorted(set(y_class))},
                "channels": CHANNELS,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "p300_window_ms": [int(P300_WIN_S[0] * 1000), int(P300_WIN_S[1] * 1000)],
            },
            sort_keys=True,
        )
    )

    test_trials = make_oddball_trials(dataset, args.trials, rng)
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    emitted = 0
    stats_by_class = {
        cfg["label"]: {"events": 0, "correct": 0, "p300_detected": 0, "p300_strength_sum": 0.0, "confidence_sum": 0.0}
        for cfg in CATEGORY_CONFIG.values()
    }
    predicted_counts = {cfg["label"]: 0 for cfg in CATEGORY_CONFIG.values()}
    with output_path.open("w", encoding="utf-8") as fh:
        for idx, stim in enumerate(test_trials, start=1):
            maybe_show_image(stim, args.display, args.fixation_s, args.image_s, args.blank_s)
            ts = time.time()
            event = predict_event(stim, ts, source, p300_model, class_model)
            if event is None:
                continue
            emitted += 1
            correct += int(event.predicted_class == event.true_class)
            class_stats = stats_by_class[event.true_class]
            class_stats["events"] += 1
            class_stats["correct"] += int(event.predicted_class == event.true_class)
            class_stats["p300_detected"] += int(event.p300_detected)
            class_stats["p300_strength_sum"] += event.p300_strength
            class_stats["confidence_sum"] += event.confidence
            predicted_counts[event.predicted_class] = predicted_counts.get(event.predicted_class, 0) + 1
            line = event.to_json()
            fh.write(line + "\n")
            print(line)
            if args.dry_run and idx >= args.trials:
                break

    accuracy = correct / emitted if emitted else 0.0
    class_summary = {}
    for label, values in stats_by_class.items():
        events = values["events"]
        class_summary[label] = {
            "events": events,
            "accuracy": round(values["correct"] / events, 3) if events else 0.0,
            "p300_detect_rate": round(values["p300_detected"] / events, 3) if events else 0.0,
            "mean_p300_strength": round(values["p300_strength_sum"] / events, 3) if events else 0.0,
            "mean_confidence": round(values["confidence_sum"] / events, 3) if events else 0.0,
        }
    print(
        json.dumps(
            {
                "type": "session_complete",
                "events": emitted,
                "accuracy": round(accuracy, 3),
                "predicted_counts": predicted_counts,
                "class_summary": class_summary,
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock EEG P300 salience classifier for fire-scene images.")
    parser.add_argument("--repo-root", default=".", help="Repository root containing images/ category folders.")
    parser.add_argument("--calibration-trials", type=int, default=180, help="Trials used to train per-user demo models.")
    parser.add_argument("--trials", type=int, default=45, help="Real-time inference trials to run.")
    parser.add_argument("--output", default="outputs/eeg_salience_events.jsonl", help="JSONL event output path.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible demo sessions.")
    parser.add_argument("--display", action="store_true", help="Use pygame to display fixation/image/blank stimuli.")
    parser.add_argument("--dry-run", action="store_true", help="Headless fast run for smoke tests.")
    parser.add_argument("--fixation-s", type=float, default=0.5, help="Fixation cross duration.")
    parser.add_argument("--image-s", type=float, default=0.4, help="Image stimulus duration.")
    parser.add_argument("--blank-s", type=float, default=0.8, help="Blank screen duration.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_session(parse_args(sys.argv[1:])))
