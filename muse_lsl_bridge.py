"""
Muse 2 LSL bridge for the NeuroRSVP collection web app.

Run `muselsl stream` first so the Muse 2 EEG stream is available over LSL, then:

  python muse_lsl_bridge.py

The bridge reads LSL EEG chunks, converts LSL timestamps into Unix epoch
milliseconds so they line up with browser stimulus timestamps, and posts packets
to the Node server. It also computes rolling relative bandpowers for
delta/theta/alpha/beta/gamma using the same preprocessing pattern from the older
Muse talk/silent project.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from collections import deque
from typing import Any

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop
from scipy.signal import welch


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def post_json(url: str, payload: dict[str, Any], timeout: float = 2.0) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw) if raw else {}


def get_channel_labels(info, n_channels: int) -> list[str]:
    labels = []
    try:
        ch = info.desc().child("channels").child("channel")
        for idx in range(n_channels):
            label = ch.child_value("label")
            labels.append(label if label else f"Ch{idx + 1}")
            ch = ch.next_sibling()
    except Exception:
        labels = [f"Ch{idx + 1}" for idx in range(n_channels)]

    if len(labels) != n_channels:
        labels = [f"Ch{idx + 1}" for idx in range(n_channels)]
    return labels


def compute_relative_bandpowers(signal_1d: np.ndarray, fs: float) -> dict[str, float]:
    if signal_1d.size < int(1.5 * fs):
        return {band: 0.0 for band in BANDS}

    freqs, psd = welch(
        signal_1d,
        fs=fs,
        nperseg=min(signal_1d.size, int(2 * fs)),
        detrend="constant",
    )
    total_mask = (freqs >= 1.0) & (freqs <= min(45.0, fs / 2.0))
    total_power = np.trapezoid(psd[total_mask], freqs[total_mask])

    out = {}
    for band_name, (fmin, fmax) in BANDS.items():
        mask = (freqs >= fmin) & (freqs <= min(fmax, fs / 2.0))
        power = np.trapezoid(psd[mask], freqs[mask])
        out[band_name] = float(power / total_power) if total_power > 0 else 0.0
    return out


def compute_average_bands(buffers: list[deque[float]], fs: float) -> dict[str, float]:
    per_channel = []
    for buf in buffers:
        signal = np.asarray(buf, dtype=np.float64)
        per_channel.append(compute_relative_bandpowers(signal, fs))
    return {
        band: float(np.mean([row[band] for row in per_channel])) if per_channel else 0.0
        for band in BANDS
    }


def resolve_stream(stream_type: str, timeout: float) -> Any:
    print(f"Looking for LSL stream type={stream_type!r}. Start `muselsl stream` first if needed.")
    streams = resolve_byprop("type", stream_type, timeout=timeout)
    if not streams:
        raise RuntimeError("No EEG LSL stream found. Make sure your Muse is connected and `muselsl stream` is running.")
    return streams[0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Bridge Muse 2 LSL EEG into the NeuroRSVP web app.")
    parser.add_argument("--server", default="http://127.0.0.1:3000", help="Node server base URL.")
    parser.add_argument("--stream-type", default="EEG", help="LSL stream type to resolve.")
    parser.add_argument("--resolve-timeout", type=float, default=10.0, help="Seconds to wait for LSL stream.")
    parser.add_argument("--max-samples", type=int, default=32, help="Samples to pull per LSL chunk.")
    parser.add_argument("--buffer-seconds", type=float, default=5.0, help="Rolling window for live bandpowers.")
    parser.add_argument("--bands-interval", type=float, default=0.5, help="Seconds between bandpower updates.")
    args = parser.parse_args()

    stream = resolve_stream(args.stream_type, args.resolve_timeout)
    inlet = StreamInlet(stream, max_buflen=30)
    info = inlet.info()

    fs = float(info.nominal_srate()) if info.nominal_srate() and info.nominal_srate() > 0 else 256.0
    n_channels = info.channel_count()
    channel_labels = get_channel_labels(info, n_channels)
    clock_offset_ms = (time.time() - local_clock()) * 1000.0

    print(f"Connected to Muse LSL stream: fs={fs:.1f} Hz, channels={n_channels}, labels={channel_labels}")
    print(f"Posting EEG packets to {args.server}/api/muse/eeg")
    print("Press Ctrl+C to stop.\n")

    buffer_len = max(1, int(args.buffer_seconds * fs))
    buffers = [deque(maxlen=buffer_len) for _ in range(n_channels)]
    last_bands_post = 0.0
    packets = 0
    samples_seen = 0

    try:
      while True:
        chunk, timestamps = inlet.pull_chunk(timeout=0.2, max_samples=args.max_samples)
        if not chunk:
            continue

        arr = np.asarray(chunk, dtype=np.float64)
        if arr.ndim != 2:
            continue

        usable_channels = min(n_channels, arr.shape[1])
        arr = arr[:, :usable_channels]
        channels = channel_labels[:usable_channels]

        for row in arr:
            for channel_idx, value in enumerate(row):
                buffers[channel_idx].append(float(value))

        if timestamps:
            start_time_ms = float(timestamps[0] * 1000.0 + clock_offset_ms)
        else:
            start_time_ms = time.time() * 1000.0

        payload = {
            "data": arr.tolist(),
            "startTime": start_time_ms,
            "sampleRate": fs,
            "channels": channels,
            "source": "muse2-lsl",
        }
        post_json(f"{args.server}/api/muse/eeg", payload)
        packets += 1
        samples_seen += arr.shape[0]

        now = time.time()
        if now - last_bands_post >= args.bands_interval and len(buffers[0]) >= int(1.5 * fs):
            band_payload = {
                "source": "muse2-lsl",
                "sampleRate": fs,
                "channels": channels,
                "bands": compute_average_bands(buffers[:usable_channels], fs),
                "windowSeconds": args.buffer_seconds,
            }
            post_json(f"{args.server}/api/muse/bands", band_payload)
            last_bands_post = now

        if packets % 50 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] packets={packets} samples={samples_seen}")

    except KeyboardInterrupt:
        print("\nStopped Muse bridge.")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not post to Node server at {args.server}. Is `npm start` running? {exc}") from exc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
