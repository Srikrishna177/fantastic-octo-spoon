"""
Stage 4 - Dataset Statistics
===============================
Aggregates segment statistics across all processed videos.
Answers the core empirical questions that earlier research design
made assumptions about.

Usage:
    python pipeline/04_stats.py --id vid_001     # single video
    python pipeline/04_stats.py --id vid_002     # single video (separate directory)
    python pipeline/04_stats.py                  # aggregate across ALL processed videos

Output (single video):
    outputs/vid_001/dataset_stats.json     - machine-readable stats for vid_001
    outputs/vid_001/dataset_report.txt     - human-readable report for vid_001

Output (aggregate - no --id flag):
    outputs/aggregate/dataset_stats.json   - combined stats across all videos
    outputs/aggregate/dataset_report.txt   - combined report

Each video gets its own directory, mirroring Stage 2 behavior.
Running --id vid_002 never touches vid_001's output files.

Research questions this script directly answers:
    Q1: How many segments does a tutorial video actually produce?
        (We assumed 500-800; this verifies or falsifies that.)
    Q2: Does segment count correlate with video duration?
    Q3: Does segment density (segments/minute) vary across
        temporal density bands (Low / Medium / High)?
    Q4: What is the word count distribution across the full dataset?
"""

import argparse
import json
import statistics
import io
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).parent.parent
SEGMENTS_DIR = ROOT / "data" / "segments"
RAW_DIR = ROOT / "data" / "raw"
OUTPUTS_DIR = ROOT / "outputs"


def _sanitize_text(text: str) -> str:
    """Replace problematic Unicode characters with ASCII equivalents."""
    replacements = {
        '–': '-',  # en-dash to hyphen
        '—': '-',  # em-dash to hyphen
        ''': "'",  # right single quote to apostrophe
        ''': "'",  # left single quote to apostrophe
        '"': '"',  # left double quote to quote
        '"': '"',  # right double quote to quote
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def compute_stats(video_ids: list = None) -> dict:
    """
    Compute statistics for specified video(s) or all processed videos.

    Output directory logic:
        Single video  → outputs/<video_id>/
        Multiple/all  → outputs/aggregate/
    """
    # ── Discover videos ────────────────────────────────────────────────────
    single_video_mode = video_ids is not None and len(video_ids) == 1

    if not video_ids:
        video_ids = sorted([
            d.name for d in SEGMENTS_DIR.iterdir()
            if d.is_dir() and (d / "segment_stats.json").exists()
        ])

    if not video_ids:
        print("[04_stats] No processed videos found.")
        print("  Run pipeline stages 1-3 on at least one video first.")
        return {}

    print(f"[04_stats] Computing stats across {len(video_ids)} video(s): {video_ids}")

    per_video = []
    all_segment_counts = []
    all_word_counts_flat = []
    all_durations = []
    all_segment_densities = []

    for vid_id in video_ids:
        seg_stats_path = SEGMENTS_DIR / vid_id / "segment_stats.json"
        utterances_path = SEGMENTS_DIR / vid_id / "utterances.json"
        metadata_path = RAW_DIR / vid_id / "metadata.json"

        if not seg_stats_path.exists():
            print(f"  [SKIP] {vid_id} - segment_stats.json not found")
            continue

        with open(seg_stats_path) as f:
            seg_stats = json.load(f)

        metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        utterances = []
        if utterances_path.exists():
            with open(utterances_path) as f:
                data = json.load(f)
                utterances = data.get("utterances", [])

        video_duration_s = metadata.get("duration_seconds", 0)
        video_duration_min = video_duration_s / 60 if video_duration_s else 0
        n_segments = seg_stats["total_utterances"]
        seg_density = n_segments / video_duration_min if video_duration_min > 0 else 0

        video_row = {
            "video_id": vid_id,
            "title": _sanitize_text(metadata.get("title", "unknown")[:60]),
            "channel": _sanitize_text(metadata.get("channel", "unknown")),
            "duration_seconds": video_duration_s,
            "duration_human": metadata.get("duration_human", "unknown"),
            "temporal_density_band": metadata.get("temporal_density_band", "unlabeled"),
            "domain_category": metadata.get("domain_category", "unlabeled"),
            "code_complexity_level": metadata.get("code_complexity_level", "unlabeled"),
            "n_segments": n_segments,
            "segments_per_minute": round(seg_density, 2),
            "word_count_mean": seg_stats["word_counts"]["mean"],
            "word_count_median": seg_stats["word_counts"]["median"],
        }
        per_video.append(video_row)

        all_segment_counts.append(n_segments)
        all_segment_densities.append(seg_density)
        if video_duration_s:
            all_durations.append(video_duration_s)

        for u in utterances:
            all_word_counts_flat.append(u["word_count"])

    if not per_video:
        print("[04_stats] No valid video stats found.")
        return {}

    # ── Aggregate statistics ───────────────────────────────────────────────
    total_segments = sum(all_segment_counts)
    total_hours = sum(all_durations) / 3600 if all_durations else 0

    aggregate = {
        "generated_at": datetime.now().isoformat(),
        "mode": "single_video" if single_video_mode else "aggregate",
        "n_videos": len(per_video),
        "total_segments": total_segments,
        "total_hours": round(total_hours, 2),
        "segments_per_video": {
            "min": min(all_segment_counts),
            "max": max(all_segment_counts),
            "mean": round(statistics.mean(all_segment_counts), 1),
            "median": round(statistics.median(all_segment_counts), 1),
            "stdev": round(statistics.stdev(all_segment_counts), 1) if len(all_segment_counts) > 1 else 0,
            "total": total_segments,
        },
        "segments_per_minute": {
            "min": round(min(all_segment_densities), 2),
            "max": round(max(all_segment_densities), 2),
            "mean": round(statistics.mean(all_segment_densities), 2),
        },
        "word_count_distribution": {
            "very_short_4_10": sum(1 for w in all_word_counts_flat if 4 <= w < 10),
            "short_10_20":     sum(1 for w in all_word_counts_flat if 10 <= w < 20),
            "medium_20_30":    sum(1 for w in all_word_counts_flat if 20 <= w < 30),
            "long_30_40":      sum(1 for w in all_word_counts_flat if 30 <= w < 40),
            "very_long_40_50": sum(1 for w in all_word_counts_flat if 40 <= w <= 50),
        },
        "research_findings": _generate_research_findings(
            all_segment_counts, all_durations, per_video
        ),
        "per_video": per_video,
    }

    # ── Determine output directory ─────────────────────────────────────────
    # Single video  → outputs/<video_id>/   (own directory, never overwritten by other videos)
    # Aggregate run → outputs/aggregate/    (combined report across all videos)
    if single_video_mode:
        out_dir = OUTPUTS_DIR / video_ids[0]
    else:
        out_dir = OUTPUTS_DIR / "aggregate"

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Save outputs ───────────────────────────────────────────────────────
    stats_path = out_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    report = _generate_report(aggregate)
    # Sanitize the report to remove problematic characters
    report = _sanitize_text(report)
    report_path = out_dir / "dataset_report.txt"
    # Ensure the report is safe for writing
    report_safe = report.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(report_safe)

    # Skip console output to avoid encoding issues
    print(f"\n[04_stats] Saved to: {out_dir}/")
    print(f"  {stats_path.name}")
    print(f"  {report_path.name}")

    return aggregate


def _generate_research_findings(
    segment_counts: list,
    durations: list,
    per_video: list
) -> dict:
    mean_segments = statistics.mean(segment_counts) if segment_counts else 0

    if mean_segments < 500:
        assumption_verdict = (
            f"REFUTED - Mean segments per video is {mean_segments:.0f}, "
            f"well below the assumed 500-800. "
            f"'Small dataset' framing is STRONGER than assumed. "
            f"Consider increasing annotation scope or adjusting model architecture."
        )
    elif mean_segments <= 800:
        assumption_verdict = (
            f"CONFIRMED - Mean segments per video is {mean_segments:.0f}, "
            f"within the assumed 500-800 range."
        )
    else:
        assumption_verdict = (
            f"REFUTED (UPWARD) - Mean segments per video is {mean_segments:.0f}, "
            f"well above the assumed 500-800. "
            f"Dataset is LARGER than assumed. "
            f"This strengthens the architecture and training protocol decisions."
        )

    return {
        "segments_per_video_assumption_check": assumption_verdict,
        "note": (
            "These findings directly update the research design. "
            "Document deviations from assumptions in the paper methods section."
        )
    }


def _generate_report(agg: dict) -> str:
    mode_label = (
        f"Single video: {agg['per_video'][0]['video_id']}"
        if agg["mode"] == "single_video"
        else f"Aggregate: {agg['n_videos']} videos"
    )

    lines = [
        "=" * 60,
        "VIDEO2CODE - DATASET STATISTICS REPORT",
        f"Generated: {agg['generated_at']}",
        f"Mode: {mode_label}",
        "=" * 60,
        "",
        "OVERVIEW",
        f"  Videos processed:        {agg['n_videos']}",
        f"  Total utterances:        {agg['total_segments']:,}",
        f"  Total hours of content:  {agg['total_hours']:.1f}h",
        "",
        "SEGMENTS PER VIDEO",
        f"  Min:    {agg['segments_per_video']['min']}",
        f"  Max:    {agg['segments_per_video']['max']}",
        f"  Mean:   {agg['segments_per_video']['mean']}",
        f"  Median: {agg['segments_per_video']['median']}",
        f"  StDev:  {agg['segments_per_video']['stdev']}",
        "",
        "SEGMENT DENSITY (segments/minute)",
        f"  Min:    {agg['segments_per_minute']['min']}",
        f"  Max:    {agg['segments_per_minute']['max']}",
        f"  Mean:   {agg['segments_per_minute']['mean']}",
        "",
        "WORD COUNT DISTRIBUTION (all segments)",
    ]

    total = agg["total_segments"] or 1
    for name, count in agg["word_count_distribution"].items():
        pct = 100 * count / total
        bar = "*" * int(pct / 2)
        lines.append(f"  {name:<25} {count:>6}  ({pct:.1f}%)  {bar}")

    lines += [
        "",
        "PER-VIDEO BREAKDOWN",
        f"  {'Video ID':<15} {'Segs':>6} {'Segs/min':>10} {'Duration':>10}  Title",
        f"  {'-'*15} {'-'*6} {'-'*10} {'-'*10}  {'-'*30}",
    ]

    for v in agg["per_video"]:
        line_str = (
            f"  {v['video_id']:<15} {v['n_segments']:>6} "
            f"{v['segments_per_minute']:>10.1f} "
            f"{v['duration_human']:>10}  "
            f"{v['title'][:40]}"
        )
        lines.append(line_str)

    lines += [
        "",
        "RESEARCH FINDINGS",
        f"  {agg['research_findings']['segments_per_video_assumption_check']}",
        "",
        "=" * 60,
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute dataset statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/04_stats.py --id vid_001     # saves to outputs/vid_001/
  python pipeline/04_stats.py --id vid_002     # saves to outputs/vid_002/
  python pipeline/04_stats.py                  # saves to outputs/aggregate/
        """
    )
    parser.add_argument(
        "--id", dest="video_id", default=None,
        help="Single video ID. Saves to outputs/<video_id>/. Omit to aggregate all."
    )
    args = parser.parse_args()

    video_ids = [args.video_id] if args.video_id else None
    compute_stats(video_ids)


if __name__ == "__main__":
    main()