"""
Stage 3 — Utterance Segmentation
===================================
Converts Whisper's sentence-level segments into utterance-level segments
suitable for annotation and model training.

An utterance is a pause-bounded unit of speech: 10-40 words, defined by
silence gaps >= PAUSE_THRESHOLD_MS between words.

Usage:
    python pipeline/03_segment.py --id vid_001
    python pipeline/03_segment.py --id vid_001 --pause 600  # 600ms pause threshold

Output:
    data/segments/vid_001/utterances.json     — all utterances with metadata
    data/segments/vid_001/utterances.csv      — flat CSV for annotation tools
    data/segments/vid_001/segment_stats.json  — empirical statistics

Research purpose:
    This script answers the critical empirical question:
    "How many utterance-level segments does a real tutorial video produce?"
    
    The answer directly determines:
    - Dataset size estimates
    - Label imbalance ratios (once annotated)
    - Whether our "small dataset" framing is correct or not

Design decisions:
    PAUSE_THRESHOLD_MS: 500ms default.
        Too low  → over-segments (single sentences split mid-thought)
        Too high → under-segments (multiple distinct utterances merged)
        This threshold is a hyperparameter. We record it in output metadata
        so results are reproducible.

    MAX_WORDS_PER_SEGMENT: 50 hard cap.
        Whisper sometimes produces very long segments when VAD fails.
        We hard-split these at sentence boundaries (periods, question marks).

    MIN_WORDS_PER_SEGMENT: 4 minimum.
        Very short segments (1-3 words) are typically filler ("OK so", "alright")
        and carry no intent signals. We log them as filtered rather than deleting.
"""

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).parent.parent
TRANSCRIPT_DIR = ROOT / "data" / "transcripts"
SEGMENTS_DIR = ROOT / "data" / "segments"

# ── Segmentation constants ─────────────────────────────────────────────────
DEFAULT_PAUSE_MS = 500       # silence gap that triggers a new utterance
MAX_WORDS = 50               # hard cap — force-split longer segments
MIN_WORDS = 4                # minimum to be included (not filtered)


def segment(video_id: str, pause_threshold_ms: int = DEFAULT_PAUSE_MS) -> dict:
    """
    Produce utterance-level segments from word-level transcript.
    Returns segment statistics.
    """
    # ── Load word-level transcript ─────────────────────────────────────────
    words_path = TRANSCRIPT_DIR / video_id / "transcript_words.json"
    if not words_path.exists():
        raise FileNotFoundError(
            f"Word transcript not found: {words_path}\n"
            f"Run Stage 2 first: python pipeline/02_transcribe.py --id {video_id}"
        )

    with open(words_path) as f:
        words_data = json.load(f)

    words = words_data["words"]
    if not words:
        raise ValueError(f"No words found in transcript for {video_id}")

    print(f"[03_segment] Processing {len(words)} words for {video_id}")
    print(f"  Pause threshold: {pause_threshold_ms}ms")

    # ── Phase 1: Group words into pause-bounded utterances ─────────────────
    pause_threshold_s = pause_threshold_ms / 1000.0
    raw_utterances = []
    current_group = [words[0]]

    for i in range(1, len(words)):
        prev_word = words[i - 1]
        curr_word = words[i]
        gap = curr_word["start"] - prev_word["end"]

        if gap >= pause_threshold_s:
            # Pause detected — close current group, start new one
            raw_utterances.append(current_group)
            current_group = [curr_word]
        else:
            current_group.append(curr_word)

    if current_group:
        raw_utterances.append(current_group)

    print(f"  Pause-bounded groups: {len(raw_utterances)}")

    # ── Phase 2: Apply MAX_WORDS hard split ────────────────────────────────
    split_utterances = []
    for group in raw_utterances:
        if len(group) <= MAX_WORDS:
            split_utterances.append(group)
        else:
            # Hard split at sentence boundaries within the group
            sub_groups = _split_at_sentence_boundary(group, MAX_WORDS)
            split_utterances.extend(sub_groups)

    print(f"  After MAX_WORDS split: {len(split_utterances)}")

    # ── Phase 3: Filter MIN_WORDS and build output structure ───────────────
    utterances = []
    filtered_count = 0
    seg_id = 0

    for group in split_utterances:
        text = " ".join(w["word"] for w in group).strip()
        word_count = len(group)

        if word_count < MIN_WORDS:
            filtered_count += 1
            continue

        seg_id += 1
        utterances.append({
            "segment_id": f"{video_id}_seg_{seg_id:04d}",
            "video_id": video_id,
            "segment_index": seg_id,
            "timestamp_start": _seconds_to_hms(group[0]["start"]),
            "timestamp_end": _seconds_to_hms(group[-1]["end"]),
            "timestamp_start_seconds": round(group[0]["start"], 3),
            "timestamp_end_seconds": round(group[-1]["end"], 3),
            "duration_seconds": round(group[-1]["end"] - group[0]["start"], 3),
            "transcript": text,
            "word_count": word_count,
            # Annotation fields — empty, filled by human annotators in Stage annotation
            "labels": [],
            "confidence": None,
            "flag": False,
            "flag_reason": None,
            "annotator_id": None,
            "annotation_notes": None,
        })

    print(f"  Segments after MIN_WORDS filter: {len(utterances)}")
    print(f"  Filtered out (< {MIN_WORDS} words): {filtered_count}")

    # ── Phase 4: Compute segment statistics ───────────────────────────────
    word_counts = [u["word_count"] for u in utterances]
    durations = [u["duration_seconds"] for u in utterances]

    stats = {
        "video_id": video_id,
        "pause_threshold_ms": pause_threshold_ms,
        "total_utterances": len(utterances),
        "filtered_utterances": filtered_count,
        "word_counts": {
            "min": min(word_counts),
            "max": max(word_counts),
            "mean": round(statistics.mean(word_counts), 2),
            "median": round(statistics.median(word_counts), 2),
            "stdev": round(statistics.stdev(word_counts), 2) if len(word_counts) > 1 else 0,
        },
        "duration_seconds": {
            "min": round(min(durations), 2),
            "max": round(max(durations), 2),
            "mean": round(statistics.mean(durations), 2),
            "median": round(statistics.median(durations), 2),
        },
        "word_count_distribution": _distribution_buckets(word_counts, [
            (4, 10, "very_short"),
            (10, 20, "short"),
            (20, 30, "medium"),
            (30, 40, "long"),
            (40, 51, "very_long"),
        ]),
        # Research note: this is the empirical answer to
        # "how many segments does a tutorial video produce?"
        # Compare against the 500-800 estimate made in earlier research design.
        "research_note": (
            f"This video produced {len(utterances)} utterance-level segments "
            f"using {pause_threshold_ms}ms pause threshold. "
            f"Update dataset size assumptions accordingly."
        )
    }

    # ── Save outputs ───────────────────────────────────────────────────────
    out_dir = SEGMENTS_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    utterances_path = out_dir / "utterances.json"
    csv_path = out_dir / "utterances.csv"
    stats_path = out_dir / "segment_stats.json"

    with open(utterances_path, "w") as f:
        json.dump({"video_id": video_id, "utterances": utterances}, f, indent=2)

    # CSV for annotation tools (Label Studio, spreadsheet-based annotation)
    csv_fields = [
        "segment_id", "video_id", "timestamp_start", "timestamp_end",
        "word_count", "transcript",
        "labels", "confidence", "flag", "flag_reason",
        "annotator_id", "annotation_notes"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for u in utterances:
            row = {k: u.get(k, "") for k in csv_fields}
            row["labels"] = ""  # empty — filled by annotators
            writer.writerow(row)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[03_segment] Saved:")
    print(f"  {utterances_path}")
    print(f"  {csv_path}")
    print(f"  {stats_path}")

    # Print key statistics to terminal
    print(f"\n{'='*50}")
    print(f"SEGMENT STATISTICS - {video_id}")
    print(f"{'='*50}")
    print(f"  Total utterances:     {stats['total_utterances']}")
    print(f"  Word count (mean):    {stats['word_counts']['mean']}")
    print(f"  Word count (median):  {stats['word_counts']['median']}")
    print(f"  Word count (range):   {stats['word_counts']['min']}-{stats['word_counts']['max']}")
    print(f"  Duration (mean):      {stats['duration_seconds']['mean']}s")
    print(f"  Distribution:")
    for bucket_name, count in stats["word_count_distribution"].items():
        pct = 100 * count / stats["total_utterances"] if stats["total_utterances"] > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"    {bucket_name:<15} {count:>5}  ({pct:.1f}%)  {bar}")
    print(f"{'='*50}")
    print(f"\n  ⚠ RESEARCH NOTE: {stats['research_note']}")

    return stats


def _split_at_sentence_boundary(words: list, max_words: int) -> list:
    """
    Split a word group that exceeds MAX_WORDS at sentence boundary punctuation.
    Falls back to hard split at max_words if no boundary found.
    """
    sentence_end_pattern = re.compile(r'[.!?]$')
    groups = []
    current = []

    for w in words:
        current.append(w)
        if len(current) >= max_words // 2 and sentence_end_pattern.search(w["word"]):
            groups.append(current)
            current = []

    if current:
        if groups and len(current) < MIN_WORDS:
            # Merge tiny tail into last group rather than creating a tiny segment
            groups[-1].extend(current)
        else:
            groups.append(current)

    # Final safety: hard split anything still over MAX_WORDS
    final = []
    for g in groups:
        if len(g) <= MAX_WORDS:
            final.append(g)
        else:
            for i in range(0, len(g), MAX_WORDS):
                chunk = g[i:i + MAX_WORDS]
                if chunk:
                    final.append(chunk)

    return final


def _distribution_buckets(values: list, buckets: list) -> dict:
    result = {}
    for low, high, name in buckets:
        result[name] = sum(1 for v in values if low <= v < high)
    return result


def _seconds_to_hms(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Segment transcript into utterances")
    parser.add_argument("--id", required=True, dest="video_id")
    parser.add_argument("--pause", type=int, default=DEFAULT_PAUSE_MS,
                        dest="pause_threshold_ms",
                        help=f"Silence gap in ms that triggers a new utterance (default: {DEFAULT_PAUSE_MS})")
    args = parser.parse_args()

    segment(args.video_id, args.pause_threshold_ms)


if __name__ == "__main__":
    main()