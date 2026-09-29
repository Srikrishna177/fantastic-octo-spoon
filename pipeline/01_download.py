# -*- coding: utf-8 -*-
"""
Stage 1 - Video Downloader
===========================
Downloads audio from a YouTube tutorial video and saves metadata.

Usage:
    python pipeline/01_download.py --url "https://youtube.com/watch?v=..." --id vid_001

Output:
    data/raw/vid_001/audio.wav
    data/raw/vid_001/metadata.json

Dependencies:
    yt-dlp, ffmpeg
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw"


def download_video(url: str, video_id: str) -> dict:
    """
    Download audio from a YouTube URL using yt-dlp.
    Extracts audio to WAV format (required by Whisper).

    Returns metadata dict.
    """
    out_dir = DATA_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_path = out_dir / "audio.wav"
    metadata_path = out_dir / "metadata.json"

    # ── Step 1: Fetch video metadata without downloading ──────────────────
    print(f"[01_download] Fetching metadata for {video_id}...")
    meta_cmd = [
        "yt-dlp",
        "--dump-json",
        "--no-playlist",
        url
    ]

    result = subprocess.run(meta_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] yt-dlp metadata fetch failed:\n{result.stderr}")
        sys.exit(1)

    raw_meta = json.loads(result.stdout)

    metadata = {
        "video_id": video_id,
        "url": url,
        "title": raw_meta.get("title", "unknown"),
        "channel": raw_meta.get("channel", raw_meta.get("uploader", "unknown")),
        "duration_seconds": raw_meta.get("duration", 0),
        "duration_human": _seconds_to_hms(raw_meta.get("duration", 0)),
        "upload_date": raw_meta.get("upload_date", "unknown"),
        "view_count": raw_meta.get("view_count", 0),
        "description_excerpt": (raw_meta.get("description") or "")[:500],
        "downloaded_at": datetime.utcnow().isoformat(),
        "audio_path": str(audio_path),
        # Dataset selection dimension annotations (fill manually after download)
        "temporal_density_band": None,       # Low / Medium / High
        "domain_category": None,             # frontend / backend / fullstack / devops / data
        "duration_band": None,               # Short / Medium / Long
        "instructor_speech_style": None,     # native / non-native / scripted / conversational
        "code_complexity_level": None,       # Beginner / Intermediate / Advanced
        "selection_justification": None,     # Why this video fills a gap in the coverage matrix
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[01_download] Metadata saved: {metadata_path}")
    print(f"  Title:    {metadata['title']}")
    print(f"  Channel:  {metadata['channel']}")
    print(f"  Duration: {metadata['duration_human']}")

    # ── Step 2: Download and convert audio to WAV ─────────────────────────
    if audio_path.exists():
        print(f"[01_download] Audio already exists, skipping download: {audio_path}")
        return metadata

    print(f"[01_download] Downloading audio...")
    download_cmd = [
        "yt-dlp",
        "--no-playlist",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",          # best quality
        "--postprocessor-args", "-ar 16000 -ac 1",  # 16kHz mono (Whisper requirement)
        "--output", str(out_dir / "audio.%(ext)s"),
        url
    ]

    result = subprocess.run(download_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] yt-dlp download failed:\n{result.stderr}")
        sys.exit(1)

    # yt-dlp names file audio.wav but verify
    if not audio_path.exists():
        # Check for any wav file
        wav_files = list(out_dir.glob("*.wav"))
        if wav_files:
            wav_files[0].rename(audio_path)
        else:
            print(f"[ERROR] No WAV file found in {out_dir}")
            sys.exit(1)

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"[01_download] Audio saved: {audio_path} ({file_size_mb:.1f} MB)")

    return metadata


def _seconds_to_hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Download tutorial video audio")
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--id", required=True, dest="video_id",
                        help="Video ID (e.g. vid_001) — used for all downstream file naming")
    args = parser.parse_args()

    metadata = download_video(args.url, args.video_id)
    print(f"\n[01_download] Done. Fill in dataset selection dimensions in:")
    print(f"  {DATA_DIR / args.video_id / 'metadata.json'}")


if __name__ == "__main__":
    main()