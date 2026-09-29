"""
Stage 5 — Inter-Annotator Agreement (IAA)
==========================================
Computes Krippendorff's Alpha per intent signal category
from multiple annotators' CSV files.

Self-contained: no external dependencies beyond Python stdlib.

Usage:
    python pipeline/05_iaa.py --files ann1.csv ann2.csv ann3.csv
    python pipeline/05_iaa.py --files ann1.csv ann2.csv --labels ADS TDS ECS

Output:
    outputs/iaa_report.json   — per-label alpha scores + verdict
    outputs/iaa_report.txt    — human-readable report

IAA Targets (from annotation guide):
    CAS  >= 0.80  (low difficulty — strong lexical markers)
    SCS  >= 0.78
    ECS  >= 0.75
    TDS  >= 0.72
    PMS  >= 0.68
    NSS  >= 0.67
    ADS  >= 0.67

CSV format expected:
    segment_id, labels, annotator_id, ...
    where 'labels' is comma-separated: "ADS,CAS" or "" for NULL
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from datetime import datetime


ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = ROOT / "outputs"

ALL_LABELS = ["ADS", "TDS", "ECS", "NSS", "CAS", "SCS", "PMS"]

# IAA targets from annotation guide
# These are research design targets — not empirically validated baselines.
IAA_TARGETS = {
    "ADS": 0.67,
    "TDS": 0.72,
    "ECS": 0.75,
    "NSS": 0.67,
    "CAS": 0.80,
    "SCS": 0.78,
    "PMS": 0.68,
}


def krippendorff_alpha_nominal(reliability_data: list) -> float:
    """
    Compute Krippendorff's Alpha for binary nominal data.
    No external dependencies.

    Args:
        reliability_data: list of lists — rows=annotators, cols=units
                          Values: 0, 1, or float('nan') for missing

    Returns:
        float alpha in [-1, 1], or float('nan') if undefined

    Formula: alpha = 1 - (D_o / D_e)
    where D_o = observed disagreement, D_e = expected disagreement.

    Reference: Krippendorff (2004) Content Analysis, 2nd ed., p. 221-250
    The binary nominal case simplifies the general formula significantly.
    """
    n_annotators = len(reliability_data)
    if n_annotators < 2:
        return float("nan")
    n_units = len(reliability_data[0]) if reliability_data else 0
    if n_units == 0:
        return float("nan")

    # Build coincidence table
    # For each unit, collect all pairs of values from different annotators
    coincidences = []

    for unit_idx in range(n_units):
        unit_values = []
        for ann in range(n_annotators):
            v = reliability_data[ann][unit_idx]
            if not math.isnan(v):
                unit_values.append(v)

        m_u = len(unit_values)
        if m_u < 2:
            continue

        # Each pair contributes 1/(m_u - 1) to avoid over-counting
        # (standard Krippendorff pairing weight)
        weight = 1.0 / (m_u - 1)
        for i in range(m_u):
            for j in range(i + 1, m_u):
                coincidences.append((unit_values[i], unit_values[j], weight))

    if not coincidences:
        return float("nan")

    total_weight = sum(w for _, _, w in coincidences)
    if total_weight == 0:
        return float("nan")

    # Observed disagreement (weighted)
    D_o = sum(w for v1, v2, w in coincidences if v1 != v2) / total_weight

    # Expected disagreement — from marginal distribution
    # Collect all non-missing values (with unit weights for marginal)
    all_vals = []
    for unit_idx in range(n_units):
        unit_values = []
        for ann in range(n_annotators):
            v = reliability_data[ann][unit_idx]
            if not math.isnan(v):
                unit_values.append(v)
        m_u = len(unit_values)
        if m_u < 2:
            continue
        for v in unit_values:
            all_vals.append(v)

    total_n = len(all_vals)
    if total_n < 2:
        return float("nan")

    n_0 = sum(1 for v in all_vals if v == 0)
    n_1 = sum(1 for v in all_vals if v == 1)

    # For binary nominal: D_e = 2 * p(0) * p(1)
    D_e = 2.0 * (n_0 / total_n) * (n_1 / total_n)

    if D_e == 0:
        return float("nan")  # No variance

    return 1.0 - (D_o / D_e)


def compute_iaa(annotation_files: list, labels: list = None) -> dict:
    """
    Compute Krippendorff's Alpha per label from annotator CSV files.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    labels_to_check = labels or ALL_LABELS

    # ── Load all annotations ───────────────────────────────────────────────
    annotations = defaultdict(dict)
    annotator_ids = set()

    for filepath in annotation_files:
        path = Path(filepath)
        if not path.exists():
            print(f"[WARN] File not found: {filepath}")
            continue

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seg_id = row["segment_id"].strip()
                ann_id = row.get("annotator_id", path.stem).strip()
                raw_labels = row.get("labels", "").strip()

                label_set = set()
                if raw_labels:
                    for lbl in raw_labels.split(","):
                        lbl = lbl.strip().upper()
                        if lbl in ALL_LABELS:
                            label_set.add(lbl)

                annotations[seg_id][ann_id] = label_set
                annotator_ids.add(ann_id)

    if not annotations:
        raise ValueError("No annotations loaded. Check CSV format.")

    annotator_list = sorted(annotator_ids)
    segment_ids = sorted(annotations.keys())

    print(f"[05_iaa] Loaded annotations:")
    print(f"  Segments:   {len(segment_ids)}")
    print(f"  Annotators: {len(annotator_list)} — {annotator_list}")

    # ── Filter to segments annotated by ALL annotators ─────────────────────
    complete_segments = [
        seg_id for seg_id in segment_ids
        if all(ann_id in annotations[seg_id] for ann_id in annotator_list)
    ]
    print(f"  Complete segments (all annotators): {len(complete_segments)}")

    if len(complete_segments) < 10:
        print(f"[WARN] Only {len(complete_segments)} complete segments.")
        print("  IAA estimates unreliable with < 10 complete segments.")

    # ── Compute per-label Krippendorff's Alpha ─────────────────────────────
    results = {}

    for label in labels_to_check:
        # Reliability matrix: rows=annotators, cols=complete_segments
        # 1 = label present, 0 = absent, nan = missing
        matrix = []
        for ann_id in annotator_list:
            row = []
            for seg_id in complete_segments:
                if ann_id in annotations[seg_id]:
                    row.append(1.0 if label in annotations[seg_id][ann_id] else 0.0)
                else:
                    row.append(float("nan"))
            matrix.append(row)

        # Check variance
        all_vals = [v for row in matrix for v in row if not math.isnan(v)]
        has_variance = len(set(all_vals)) > 1 and len(all_vals) > 1

        if not has_variance:
            alpha = float("nan")
            note = "No variance — all annotators agree or no positive instances. Alpha undefined."
        else:
            alpha_raw = krippendorff_alpha_nominal(matrix)
            alpha = round(alpha_raw, 4) if not math.isnan(alpha_raw) else float("nan")
            note = None

        target = IAA_TARGETS.get(label, 0.67)
        positive_count = sum(
            1 for seg_id in complete_segments
            for ann_id in annotator_list
            if label in annotations.get(seg_id, {}).get(ann_id, set())
        )
        total_judgments = len(complete_segments) * len(annotator_list)
        positive_rate = positive_count / total_judgments if total_judgments > 0 else 0

        verdict = _verdict(alpha, target)

        results[label] = {
            "alpha": alpha if not math.isnan(alpha) else None,
            "target": target,
            "verdict": verdict,
            "positive_instances": positive_count,
            "total_judgments": total_judgments,
            "positive_rate": round(positive_rate, 4),
            "note": note,
        }

        icon = "✓" if verdict == "PASS" else ("⚠" if verdict == "BORDERLINE" else "✗")
        alpha_str = f"{alpha:.4f}" if not math.isnan(alpha) else "  N/A"
        print(f"  {label:<5} alpha={alpha_str:<8} target={target:.2f}  {icon} {verdict}")
        if note:
            print(f"         {note}")

    # ── Disagreement analysis ──────────────────────────────────────────────
    disagreements = _find_disagreements(annotations, annotator_list, complete_segments)

    # ── Build output ───────────────────────────────────────────────────────
    pass_count = sum(1 for r in results.values() if r["verdict"] == "PASS")
    output = {
        "generated_at": datetime.now().isoformat(),
        "n_annotators": len(annotator_list),
        "annotator_ids": annotator_list,
        "n_segments_total": len(segment_ids),
        "n_segments_complete": len(complete_segments),
        "labels_evaluated": labels_to_check,
        "per_label": results,
        "overall_pass_rate": round(pass_count / len(results), 4) if results else 0,
        "top_disagreements": disagreements[:20],
        "action_items": _action_items(results),
    }

    json_path = OUTPUTS_DIR / "iaa_report.json"
    txt_path = OUTPUTS_DIR / "iaa_report.txt"

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    report = _generate_report(output)
    with open(txt_path, "w") as f:
        f.write(report)

    print(f"\n  Overall pass rate: {output['overall_pass_rate']*100:.0f}%")
    print(f"\n[05_iaa] Saved:")
    print(f"  {json_path}")
    print(f"  {txt_path}")

    return output


def _verdict(alpha, target: float) -> str:
    if alpha is None or math.isnan(alpha):
        return "UNDEFINED"
    if alpha >= target:
        return "PASS"
    elif alpha >= target - 0.05:
        return "BORDERLINE"
    return "FAIL"


def _find_disagreements(annotations, annotator_list, segment_ids) -> list:
    disagreements = []
    for seg_id in segment_ids:
        if not all(ann_id in annotations[seg_id] for ann_id in annotator_list):
            continue
        total_disagreement = 0
        for label in ALL_LABELS:
            votes = [1 if label in annotations[seg_id][ann_id] else 0
                     for ann_id in annotator_list]
            majority = 1 if sum(votes) > len(votes) / 2 else 0
            total_disagreement += sum(1 for v in votes if v != majority)

        if total_disagreement > 0:
            disagreements.append({
                "segment_id": seg_id,
                "total_disagreement_score": total_disagreement,
                "annotator_labels": {
                    ann_id: sorted(annotations[seg_id][ann_id])
                    for ann_id in annotator_list
                }
            })
    return sorted(disagreements, key=lambda x: -x["total_disagreement_score"])


def _action_items(results: dict) -> list:
    items = []
    for label, r in results.items():
        if r["verdict"] == "FAIL":
            items.append({
                "label": label, "priority": "HIGH",
                "action": (
                    f"Revise {label} definition in annotation guide. "
                    f"alpha={r['alpha']}, target={r['target']}. "
                    f"Convene annotator discussion before re-annotating."
                )
            })
        elif r["verdict"] == "BORDERLINE":
            items.append({
                "label": label, "priority": "MEDIUM",
                "action": (
                    f"Review {label} edge cases. "
                    f"alpha={r['alpha']}, target={r['target']}."
                )
            })
        elif r["verdict"] == "UNDEFINED":
            items.append({
                "label": label, "priority": "INFO",
                "action": f"Add more {label} examples to calibration batch."
            })
    return items


def _generate_report(output: dict) -> str:
    lines = [
        "=" * 60,
        "VIDEO2CODE — INTER-ANNOTATOR AGREEMENT REPORT",
        f"Generated: {output['generated_at']}",
        "=" * 60,
        "",
        f"Annotators: {output['n_annotators']} — {', '.join(output['annotator_ids'])}",
        f"Segments (complete): {output['n_segments_complete']} / {output['n_segments_total']}",
        "",
        "PER-LABEL KRIPPENDORFF'S ALPHA",
        f"  {'Label':<6} {'Alpha':>8}  {'Target':>8}  {'Verdict':<12}  {'Pos.Rate':>8}",
        f"  {'-'*6} {'-'*8}  {'-'*8}  {'-'*12}  {'-'*8}",
    ]
    for label, r in output["per_label"].items():
        alpha_val = r["alpha"]
        alpha_str = f"{alpha_val:.4f}" if alpha_val is not None else "   N/A  "
        icon = "✓" if r["verdict"] == "PASS" else ("⚠" if r["verdict"] == "BORDERLINE" else "✗")
        lines.append(
            f"  {label:<6} {alpha_str:>8}  {r['target']:>8.2f}  "
            f"{icon} {r['verdict']:<10}  {r['positive_rate']:>8.2%}"
        )
    lines += [
        "",
        f"Overall pass rate: {output['overall_pass_rate']*100:.0f}%",
        "",
        "ACTION ITEMS",
    ]
    if output["action_items"]:
        for item in output["action_items"]:
            lines.append(f"  [{item['priority']}] {item['label']}: {item['action']}")
    else:
        lines.append("  All labels meet targets. Proceed to full annotation.")

    if output["top_disagreements"]:
        lines += ["", "TOP DISAGREED SEGMENTS (review in calibration)"]
        for d in output["top_disagreements"][:10]:
            ann_summary = " | ".join(
                f"{ann}: {','.join(lbls) or 'NULL'}"
                for ann, lbls in d["annotator_labels"].items()
            )
            lines.append(f"  {d['segment_id']:<30} score={d['total_disagreement_score']}  {ann_summary}")
    lines += ["", "=" * 60]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compute inter-annotator agreement")
    parser.add_argument("--files", nargs="+", required=True,
                        help="Annotation CSV files (one per annotator)")
    parser.add_argument("--labels", nargs="*", default=None, choices=ALL_LABELS)
    args = parser.parse_args()
    compute_iaa(args.files, args.labels)


if __name__ == "__main__":
    main()