#!/usr/bin/env python3
"""
extract_gps.py – Extract GPS data from GoPro MAX videos (GPMF telemetry).

GoPro MAX embeds GPS telemetry using the GPMF (GoPro Metadata Format) inside
the video container.  This script locates the telemetry stream, parses the
binary GPMF data, and exports the GPS track as CSV and/or GPX.

Requirements: ffmpeg / ffprobe must be on PATH.

Usage
-----
    # Single file → CSV + GPX in the same directory
    python extract_gps.py GS010001.360

    # Multiple files
    python extract_gps.py GS010001.360 GS010002.360

    # Entire folder, CSV only
    python extract_gps.py /path/to/videos/ --format csv

    # Custom output directory
    python extract_gps.py GS010001.360 --output /tmp/gps_out --format both
"""

import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# GPMF binary parser
# ──────────────────────────────────────────────────────────────

def _unpack_scal(type_char: int, payload: bytes):
    """Return a numeric scale from a SCAL payload."""
    fmt_map = {
        ord('b'): ('>b', 1),
        ord('B'): ('>B', 1),
        ord('s'): ('>h', 2),
        ord('S'): ('>H', 2),
        ord('l'): ('>i', 4),
        ord('L'): ('>I', 4),
    }
    info = fmt_map.get(type_char)
    if info and len(payload) >= info[1]:
        return struct.unpack(info[0], payload[:info[1]])[0]
    return None


def parse_gpmf(data: bytes, depth: int = 0) -> list:
    """
    Recursively parse a GPMF binary blob and return a list of GPS dicts.

    Each dict has keys: gpsu, lat, lon, alt, speed2d, speed3d.
    """
    samples = []
    scale = None
    gpsu = None
    pos = 0

    while pos + 8 <= len(data):
        key_bytes = data[pos:pos + 4]
        type_char = data[pos + 4]          # int in Python 3
        elem_size = data[pos + 5]          # bytes per element
        repeat = (data[pos + 6] << 8) | data[pos + 7]  # big-endian uint16

        total = elem_size * repeat
        padded = (total + 3) & ~3          # round up to 4-byte boundary

        payload = data[pos + 8: pos + 8 + total]

        try:
            key = key_bytes.decode('ascii')
        except UnicodeDecodeError:
            pos += 8 + padded or 4
            continue

        # ── Nested container ──────────────────────────────────
        if type_char == 0:
            sub = parse_gpmf(payload, depth + 1)
            samples.extend(sub)

        # ── Scale factor ──────────────────────────────────────
        elif key == 'SCAL':
            scale = _unpack_scal(type_char, payload)

        # ── UTC timestamp for GPS batch ───────────────────────
        elif key == 'GPSU':
            try:
                gpsu = payload.decode('ascii', errors='replace').rstrip('\x00').strip()
            except Exception:
                gpsu = None

        # ── GPS5: lat, lon, alt, 2-D speed, 3-D speed ─────────
        elif key == 'GPS5' and scale and elem_size == 20:
            for i in range(repeat):
                off = i * 20
                if off + 20 > len(payload):
                    break
                lat, lon, alt, spd2d, spd3d = struct.unpack('>5i', payload[off: off + 20])
                samples.append({
                    'gpsu':    gpsu,
                    'lat':     lat    / scale,
                    'lon':     lon    / scale,
                    'alt':     alt    / scale,
                    'speed2d': spd2d  / scale,
                    'speed3d': spd3d  / scale,
                })

        # Advance past this KLV (minimum step = 8 to avoid infinite loop)
        step = 8 + (padded if padded > 0 else 0)
        if step == 8 and total == 0:
            step = 8  # empty KLV, advance header only
        pos += step

    return samples


# ──────────────────────────────────────────────────────────────
# ffprobe / ffmpeg helpers
# ──────────────────────────────────────────────────────────────

def find_gpmf_stream(video_path: Path) -> int | None:
    """
    Return the stream index of the GoPro GPMF telemetry track, or None.

    Looks for a stream whose codec_tag_string is 'tmcd' or 'gpmd', or whose
    tags contain 'handler_name' = 'GoPro TCD' / 'GoPro MET'.
    """
    cmd = [
        'ffprobe', '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        info = json.loads(result.stdout)
    except Exception as exc:
        print(f"  [ffprobe error] {exc}", file=sys.stderr)
        return None

    gpmf_keywords = {'gpmd', 'GoPro MET', 'GoPro TCD', 'tmcd'}

    for stream in info.get('streams', []):
        tag_str = stream.get('codec_tag_string', '').lower()
        handler = stream.get('tags', {}).get('handler_name', '')
        if tag_str in {'gpmd', 'tmcd'} or handler in {'GoPro MET', 'GoPro TCD'}:
            return stream['index']

    # Fallback: look for 'bin_data' codec (some GoPro versions)
    for stream in info.get('streams', []):
        if stream.get('codec_name') in ('bin_data', 'data'):
            handler = stream.get('tags', {}).get('handler_name', '')
            if 'gopro' in handler.lower() or 'met' in handler.lower():
                return stream['index']

    # Last resort: return the last data-type stream
    for stream in reversed(info.get('streams', [])):
        if stream.get('codec_type') == 'data':
            return stream['index']

    return None


def extract_gpmf_binary(video_path: Path, stream_index: int) -> bytes | None:
    """Dump the GPMF stream bytes using ffmpeg into a temp file and read it."""
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        'ffmpeg', '-y', '-v', 'quiet',
        '-i', str(video_path),
        '-map', f'0:{stream_index}',
        '-codec', 'copy',
        '-f', 'rawvideo',
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120, capture_output=True)
        with open(tmp_path, 'rb') as f:
            return f.read()
    except Exception as exc:
        print(f"  [ffmpeg error] {exc}", file=sys.stderr)
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ──────────────────────────────────────────────────────────────
# Timestamp helpers
# ──────────────────────────────────────────────────────────────

def parse_gpsu(gpsu: str | None) -> datetime | None:
    """
    Parse a GPSU string (YYMMDDHHMMSS.SSS) into a UTC datetime.
    Returns None on failure.
    """
    if not gpsu:
        return None
    try:
        # Format: YYMMDDHHMMSS.sss  (e.g. "230815143022.000")
        dt = datetime.strptime(gpsu[:15], '%y%m%d%H%M%S.%f')
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            dt = datetime.strptime(gpsu[:12], '%y%m%d%H%M%S')
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def interpolate_timestamps(samples: list, fps: float = 18.0) -> list:
    """
    Fill in per-sample datetime objects.  GoPro GPS runs at ~18 Hz.
    Samples in the same GPSU group share a start time; we interpolate
    within each group using the given GPS Hz rate.
    """
    if not samples:
        return samples

    # Group by gpsu string
    groups: list[list] = []
    current_gpsu = object()  # sentinel
    for s in samples:
        if s['gpsu'] != current_gpsu:
            groups.append([])
            current_gpsu = s['gpsu']
        groups[-1].append(s)

    out = []
    for group in groups:
        base_dt = parse_gpsu(group[0]['gpsu'])
        for i, s in enumerate(group):
            s = dict(s)
            if base_dt is not None:
                s['datetime'] = base_dt + timedelta(seconds=i / fps)
            else:
                s['datetime'] = None
            out.append(s)

    return out


# ──────────────────────────────────────────────────────────────
# Exporters
# ──────────────────────────────────────────────────────────────

def write_csv(samples: list, output_path: Path):
    fieldnames = ['datetime_utc', 'latitude', 'longitude', 'altitude_m',
                  'speed2d_ms', 'speed3d_ms']
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            dt = s.get('datetime')
            writer.writerow({
                'datetime_utc': dt.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z' if dt else '',
                'latitude':     f"{s['lat']:.7f}",
                'longitude':    f"{s['lon']:.7f}",
                'altitude_m':   f"{s['alt']:.3f}",
                'speed2d_ms':   f"{s['speed2d']:.3f}",
                'speed3d_ms':   f"{s['speed3d']:.3f}",
            })
    print(f"  CSV  → {output_path}  ({len(samples)} points)")


def write_gpx(samples: list, output_path: Path, track_name: str = 'GoPro MAX GPS'):
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<gpx version="1.1" creator="extract_gps.py"',
        '     xmlns="http://www.topografix.com/GPX/1/1"',
        '     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '     xsi:schemaLocation="http://www.topografix.com/GPX/1/1'
        ' http://www.topografix.com/GPX/1/1/gpx.xsd">',
        f'  <trk><name>{track_name}</name><trkseg>',
    ]
    for s in samples:
        dt = s.get('datetime')
        time_tag = (
            f'<time>{dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]}Z</time>'
            if dt else ''
        )
        lines.append(
            f'    <trkpt lat="{s["lat"]:.7f}" lon="{s["lon"]:.7f}">'
            f'<ele>{s["alt"]:.3f}</ele>{time_tag}'
            f'<extensions><speed>{s["speed3d"]:.3f}</speed></extensions>'
            f'</trkpt>'
        )
    lines += ['  </trkseg></trk>', '</gpx>']

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  GPX  → {output_path}  ({len(samples)} points)")


# ──────────────────────────────────────────────────────────────
# Per-file pipeline
# ──────────────────────────────────────────────────────────────

def process_video(video_path: Path, output_dir: Path, fmt: str):
    print(f"\nProcessing: {video_path.name}")

    stream_idx = find_gpmf_stream(video_path)
    if stream_idx is None:
        print("  No GPMF telemetry stream found – skipping.")
        return

    print(f"  GPMF stream index: {stream_idx}")
    raw = extract_gpmf_binary(video_path, stream_idx)
    if not raw:
        print("  Failed to extract telemetry binary – skipping.")
        return

    print(f"  Telemetry size: {len(raw):,} bytes")
    samples = parse_gpmf(raw)

    if not samples:
        print("  No GPS samples found (GPS may have been disabled or no fix).")
        return

    samples = interpolate_timestamps(samples)

    # Filter out obviously invalid coordinates (0,0 = no fix)
    valid = [s for s in samples if not (s['lat'] == 0.0 and s['lon'] == 0.0)]
    if len(valid) < len(samples):
        print(f"  Filtered {len(samples) - len(valid)} zero-coordinate samples.")
    samples = valid

    if not samples:
        print("  All GPS samples had zero coordinates (no satellite fix).")
        return

    print(f"  GPS samples: {len(samples)}")
    lat_range = f"{min(s['lat'] for s in samples):.5f} – {max(s['lat'] for s in samples):.5f}"
    lon_range = f"{min(s['lon'] for s in samples):.5f} – {max(s['lon'] for s in samples):.5f}"
    print(f"  Latitude:  {lat_range}")
    print(f"  Longitude: {lon_range}")

    stem = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if fmt in ('csv', 'both'):
        write_csv(samples, output_dir / f"{stem}_gps.csv")
    if fmt in ('gpx', 'both'):
        write_gpx(samples, output_dir / f"{stem}_gps.gpx", track_name=stem)


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

SUPPORTED_EXT = {'.360', '.mp4', '.MP4', '.lrv', '.LRV'}


def collect_videos(paths: list[str]) -> list[Path]:
    videos = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in SUPPORTED_EXT:
                videos.extend(sorted(path.glob(f'*{ext}')))
        elif path.is_file() and path.suffix in SUPPORTED_EXT:
            videos.append(path)
        else:
            print(f"Warning: skipping '{p}' (not a supported file or directory)", file=sys.stderr)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description='Extract GPS telemetry from GoPro MAX videos (GPMF format).'
    )
    parser.add_argument(
        'inputs', nargs='+',
        help='Video file(s) or folder(s) containing GoPro MAX videos.'
    )
    parser.add_argument(
        '--output', '-o', default=None,
        help='Output directory (default: same directory as each video).'
    )
    parser.add_argument(
        '--format', '-f', choices=['csv', 'gpx', 'both'], default='both',
        help='Output format (default: both).'
    )
    args = parser.parse_args()

    videos = collect_videos(args.inputs)
    if not videos:
        print("No supported video files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} video(s) to process.")

    for video in videos:
        out_dir = Path(args.output) if args.output else video.parent
        process_video(video, out_dir, args.format)

    print("\nDone.")


if __name__ == '__main__':
    main()
