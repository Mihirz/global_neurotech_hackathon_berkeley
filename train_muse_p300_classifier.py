"""
Train a simple classifier from Muse/P300 collection sessions.

Input is the JSON saved by the collection page under collection_data/.

Example:
  python train_muse_p300_classifier.py --sessions "collection_data/*.json"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "p300Score",
    "confidence",
    "p300AmplitudeUv",
    "meanAmplitudeUv",
    "p300LatencyMs",
    "auc",
    "samples",
    "thresholdUv",
]


def flatten_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("rejected"):
        return None
    label = event.get("label")
    if label not in {"human_fire", "item_fire", "normal"}:
        return None

    row: dict[str, Any] = {
        "sessionId": event.get("sessionId"),
        "trialIndex": event.get("trialIndex"),
        "label": label,
        "imageId": event.get("imageId"),
    }
    for key in BASE_FEATURES:
        row[key] = event.get(key, 0) or 0

    band_features = event.get("bandFeatures") or {}
    for key, value in band_features.items():
        if isinstance(value, (int, float)):
            row[f"band_{key}"] = value
    return row


def load_sessions(patterns: list[str]) -> pd.DataFrame:
    paths: list[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No session JSON files matched.")

    rows = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            session = json.load(fh)
        for event in session.get("events", []):
            row = flatten_event(event)
            if row:
                rows.append(row)

    if not rows:
        raise RuntimeError("No usable non-rejected events found.")
    return pd.DataFrame(rows).fillna(0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a Muse/P300 image classifier from collection session JSON.")
    parser.add_argument("--sessions", nargs="+", default=["collection_data/*.json"], help="Session JSON paths or glob patterns.")
    parser.add_argument("--model-out", default="models/muse_p300_classifier.joblib", help="Output joblib model path.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df = load_sessions(args.sessions)
    feature_cols = [col for col in df.columns if col not in {"sessionId", "trialIndex", "label", "imageId"}]
    if len(set(df["label"])) < 2:
        raise RuntimeError("Need at least two labels/classes to train.")

    x = df[feature_cols]
    y = df["label"]
    stratify = y if y.value_counts().min() >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("Class counts:")
    print(y.value_counts())
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=sorted(y.unique())))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "labels": sorted(y.unique()),
            "source_sessions": args.sessions,
        },
        out_path,
    )
    print(f"\nSaved model to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
