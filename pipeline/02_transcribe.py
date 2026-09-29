"""
Stage 2 — ASR Transcription
=============================
Transcribes audio using faster-whisper (Whisper large-v3).
Produces word-level and segment-level timestamped transcripts.

Usage:
    python pipeline/02_transcribe.py --id vid_001
    python pipeline/02_transcribe.py --id vid_001 --model large-v3
    python pipeline/02_transcribe.py --id vid_001 --model base  # faster, less accurate

Output:
    data/transcripts/vid_001/transcript_raw.json      — raw Whisper output
    data/transcripts/vid_001/transcript_words.json    — word-level with timestamps
    data/transcripts/vid_001/transcript_segments.json — sentence-level with timestamps

Dependencies:
    faster-whisper

Research notes:
    - We use faster-whisper over openai-whisper for 4x speed + lower memory
    - Word-level timestamps are required for utterance segmentation in Stage 3
    - The VAD filter (vad_filter=True) removes non-speech segments automatically
      which reduces false positive utterances from silence / music intros
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"


def transcribe(video_id: str, model_size: str = "large-v3") -> dict:
    """
    Transcribe audio for the given video_id.
    Returns the full transcript data structure.
    """
    # ── Import here so the script gives a clear error if not installed ────
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[ERROR] faster-whisper not installed.")
        print("  Run: pip install faster-whisper")
        sys.exit(1)

    audio_path = RAW_DIR / video_id / "audio.wav"
    if not audio_path.exists():
        print(f"[ERROR] Audio not found: {audio_path}")
        print(f"  Run Stage 1 first: python pipeline/01_download.py --id {video_id}")
        sys.exit(1)

    out_dir = TRANSCRIPT_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "transcript_raw.json"
    words_path = out_dir / "transcript_words.json"
    segments_path = out_dir / "transcript_segments.json"

    if raw_path.exists():
        print(f"[02_transcribe] Transcript already exists, loading: {raw_path}")
        with open(raw_path) as f:
            return json.load(f)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"[02_transcribe] Loading Whisper model: {model_size}")
    print(f"  (First run downloads ~3GB for large-v3)")

    # Use int8 quantization for memory efficiency on consumer GPUs
    # Falls back to CPU if no GPU available
    model = WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
    )

    print(f"[02_transcribe] Transcribing: {audio_path}")
    t0 = time.time()

    # ── Run transcription ─────────────────────────────────────────────────
    # word_timestamps=True is critical for Stage 3 (utterance segmentation)
    # vad_filter=True removes silence/non-speech — reduces noise
    # beam_size=5 is the Whisper default
    segments_iter, info = model.transcribe(
        str(audio_path),
        language="en",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
    )

    # ── Collect results ───────────────────────────────────────────────────
    raw_segments = []
    all_words = []

    for seg in segments_iter:
        seg_data = {
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
            "no_speech_prob": round(seg.no_speech_prob, 4),
            "words": []
        }

        if seg.words:
            for w in seg.words:
                word_data = {
                    "word": w.word.strip(),
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 4),
                }
                seg_data["words"].append(word_data)
                all_words.append(word_data)

        raw_segments.append(seg_data)

    elapsed = time.time() - t0
    audio_duration = info.duration

    print(f"[02_transcribe] Transcription complete in {elapsed:.1f}s")
    print(f"  Audio duration: {_seconds_to_hms(int(audio_duration))}")
    print(f"  Segments: {len(raw_segments)}")
    print(f"  Total words: {len(all_words)}")
    print(f"  Real-time factor: {audio_duration / elapsed:.1f}x")

    # ── Build transcript data structure ───────────────────────────────────
    transcript_raw = {
        "video_id": video_id,
        "model": model_size,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_seconds": round(audio_duration, 2),
        "duration_human": _seconds_to_hms(int(audio_duration)),
        "total_words": len(all_words),
        "total_whisper_segments": len(raw_segments),
        "transcription_time_seconds": round(elapsed, 1),
        "segments": raw_segments,
    }

    # Word-level flat list
    transcript_words = {
        "video_id": video_id,
        "total_words": len(all_words),
        "words": all_words,
    }

    # Clean sentence-level segments (for human review)
    transcript_segments_clean = {
        "video_id": video_id,
        "total_segments": len(raw_segments),
        "segments": [
            {
                "id": s["id"],
                "start": s["start"],
                "end": s["end"],
                "text": s["text"],
            }
            for s in raw_segments
        ]
    }

    # ── Save outputs ──────────────────────────────────────────────────────
    with open(raw_path, "w") as f:
        json.dump(transcript_raw, f, indent=2)

    with open(words_path, "w") as f:
        json.dump(transcript_words, f, indent=2)

    with open(segments_path, "w") as f:
        json.dump(transcript_segments_clean, f, indent=2)

    print(f"[02_transcribe] Saved:")
    print(f"  {raw_path}")
    print(f"  {words_path}")
    print(f"  {segments_path}")

    return transcript_raw


def _seconds_to_hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe tutorial video audio")
    parser.add_argument("--id", required=True, dest="video_id")
    parser.add_argument("--model", default="large-v3",
                        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                        help="Whisper model size. large-v3 recommended for research quality.")
    args = parser.parse_args()

    transcribe(args.video_id, args.model)


if __name__ == "__main__":
    main()