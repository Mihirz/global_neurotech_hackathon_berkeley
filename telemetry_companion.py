#!/usr/bin/env python3
"""
telemetry_companion.py — DJI Mini 2 SE GPS bridge

Reads live GPS telemetry from the DJI Fly app running on an Android phone
connected via USB, then POSTs it to the NeuroRSVP server at /api/telemetry.

Requirements:
  pip install requests
  adb installed and in PATH (Android Platform Tools)
  USB debugging enabled on the Android phone

Setup:
  1. Connect Android phone to PC via USB
  2. Enable USB debugging: Settings → Developer Options → USB Debugging
  3. Pair the DJI RC-N1 controller with the phone and launch DJI Fly
  4. Arm the drone (it must be powered on and connected)
  5. Run: python3 telemetry_companion.py
  6. GPS data will stream to http://localhost:3000/api/telemetry

The DJI Fly app logs telemetry to Android logcat under the tag "FlightRecorder"
and "DJIFly". We extract lat/lon/alt/heading from those log lines.
"""

import subprocess
import re
import time
import requests
import sys
import json
from datetime import datetime

SERVER_URL = "http://localhost:3000/api/telemetry"
POLL_INTERVAL = 0.5  # seconds between logcat reads

# Regex patterns for DJI Fly logcat output
# These match the standard DJI Fly telemetry log format
PATTERNS = {
    # Full telemetry line: lat, lon, alt, heading, speed
    'full': re.compile(
        r'lat[:\s=]+(-?\d+\.\d+).*?'
        r'lon(?:g)?[:\s=]+(-?\d+\.\d+).*?'
        r'alt(?:itude)?[:\s=]+(-?\d+\.?\d*).*?'
        r'head(?:ing)?[:\s=]+(-?\d+\.?\d*)',
        re.IGNORECASE | re.DOTALL
    ),
    # Coordinate-only line (fallback)
    'coords': re.compile(
        r'(-?\d{2,3}\.\d{5,})[\s,]+(-?\d{2,3}\.\d{5,})'
    ),
    # Altitude line
    'alt': re.compile(r'altitude[:\s=]+(-?\d+\.?\d*)', re.IGNORECASE),
    # Heading line
    'heading': re.compile(r'heading[:\s=]+(-?\d+\.?\d*)', re.IGNORECASE),
    # Speed
    'speed': re.compile(r'speed[:\s=]+(-?\d+\.?\d*)', re.IGNORECASE),
}

def start_logcat():
    """Start adb logcat filtered to DJI Fly telemetry tags."""
    cmd = [
        'adb', 'logcat', '-v', 'raw',
        '-s',
        'FlightRecorder:V',
        'DJIFly:V',
        'DJISDKManager:V',
        'FlightController:V',
        'Aircraft:V',
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return proc
    except FileNotFoundError:
        print("ERROR: 'adb' not found. Install Android Platform Tools.")
        print("Download: https://developer.android.com/studio/releases/platform-tools")
        sys.exit(1)

def check_adb_device():
    """Check that a device is connected."""
    result = subprocess.run(['adb', 'devices'], capture_output=True, text=True)
    lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
    devices = [l for l in lines if '\tdevice' in l]
    if not devices:
        print("ERROR: No Android device found via ADB.")
        print("  1. Connect phone via USB")
        print("  2. Enable USB debugging (Settings → Developer Options)")
        print("  3. Accept the 'Allow USB debugging' prompt on the phone")
        sys.exit(1)
    print(f"[ADB] Device connected: {devices[0]}")
    return True

def parse_telemetry(line, state):
    """
    Parse a logcat line and update the running state dict.
    Returns updated state (may be partially filled).
    """
    # Try full pattern first
    m = PATTERNS['full'].search(line)
    if m:
        state.update({
            'lat': float(m.group(1)),
            'lon': float(m.group(2)),
            'alt': float(m.group(3)),
            'heading': float(m.group(4)),
        })
        return state

    # Fallback: extract individual values
    m = PATTERNS['coords'].search(line)
    if m:
        lat, lon = float(m.group(1)), float(m.group(2))
        # Basic sanity: valid lat/lon ranges
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            state['lat'] = lat
            state['lon'] = lon

    m = PATTERNS['alt'].search(line)
    if m:
        state['alt'] = float(m.group(1))

    m = PATTERNS['heading'].search(line)
    if m:
        state['heading'] = float(m.group(1)) % 360

    m = PATTERNS['speed'].search(line)
    if m:
        state['speed'] = float(m.group(1))

    return state

def post_telemetry(state):
    """POST telemetry to the NeuroRSVP server."""
    if state.get('lat') is None or state.get('lon') is None:
        return False
    try:
        payload = {
            'lat': state['lat'],
            'lon': state['lon'],
            'alt': state.get('alt', 0),
            'heading': state.get('heading', 0),
            'speed': state.get('speed', 0),
            'source': 'dji_mini2se',
        }
        r = requests.post(SERVER_URL, json=payload, timeout=1)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False

def main():
    print("=" * 55)
    print("  DJI Mini 2 SE Telemetry Companion")
    print(f"  Target: {SERVER_URL}")
    print("=" * 55)

    check_adb_device()

    print("[ADB] Starting logcat stream...")
    proc = start_logcat()

    state = {'lat': None, 'lon': None, 'alt': None, 'heading': None, 'speed': None}
    last_post = 0
    post_count = 0
    last_valid_gps = None

    print("[Stream] Reading telemetry. Fly the drone to see GPS data.\n")

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            state = parse_telemetry(line, state)

            # Post to server at most every POLL_INTERVAL seconds
            now = time.time()
            if state['lat'] is not None and now - last_post >= POLL_INTERVAL:
                ok = post_telemetry(state)
                if ok:
                    post_count += 1
                    ts = datetime.now().strftime('%H:%M:%S')
                    last_valid_gps = f"{state['lat']:.6f}, {state['lon']:.6f}"
                    print(
                        f"\r[{ts}] GPS: {last_valid_gps}  "
                        f"Alt: {state.get('alt', 0):.1f}m  "
                        f"Hdg: {state.get('heading', 0):.0f}°  "
                        f"({post_count} posts)   ",
                        end='', flush=True
                    )
                else:
                    print(f"\r[!] Server not reachable at {SERVER_URL}   ", end='', flush=True)
                last_post = now

    except KeyboardInterrupt:
        print(f"\n\n[Done] Posted {post_count} telemetry updates.")
        proc.terminate()

# ── Simulation mode ─────────────────────────────────────────────────────────
# Run with --simulate to generate fake GPS for testing without a drone

def simulate():
    """Simulate a drone flight path over a building for testing."""
    import math

    print("[SIMULATE] Generating fake DJI Mini 2 SE flight path")
    print(f"[SIMULATE] Posting to {SERVER_URL}\n")

    # Start point: Berkeley, CA (near hackathon venue)
    lat0, lon0 = 37.8715, -122.2596
    alt = 15.0
    heading = 0.0

    t = 0
    while True:
        # Slow orbit pattern
        lat = lat0 + 0.0002 * math.sin(t * 0.05)
        lon = lon0 + 0.0003 * math.cos(t * 0.05)
        alt = 15 + 5 * math.sin(t * 0.02)
        heading = (t * 3) % 360
        speed = 3.0

        payload = {'lat': lat, 'lon': lon, 'alt': alt, 'heading': heading, 'speed': speed, 'source': 'simulated'}
        try:
            r = requests.post(SERVER_URL, json=payload, timeout=1)
            ts = datetime.now().strftime('%H:%M:%S')
            print(f"\r[{ts}] SIM GPS: {lat:.6f}, {lon:.6f}  Alt: {alt:.1f}m  Hdg: {heading:.0f}°   ", end='', flush=True)
        except:
            print(f"\r[!] Cannot reach {SERVER_URL}   ", end='', flush=True)

        t += 1
        time.sleep(0.5)

if __name__ == '__main__':
    if '--simulate' in sys.argv:
        simulate()
    else:
        main()
