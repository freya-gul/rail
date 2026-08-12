"""Evaluate the MONAI RetinaNet detector's nodule counting against pylidc ground
truth, per the pipeline plan's Stage B (`.claude/lung_nodule_pipeline_plan.md`).

Reports two things, because "count" is meaningless without stating both the
score-threshold operating point and the consensus rule used to define a ground-truth
nodule (plan section 7's open decisions):

- A FROC-style sweep: sensitivity vs. average false-positives/scan across score
  thresholds, evaluated against two ground-truth definitions:
    * all_gt:    every pylidc annotation cluster, any number of readers (what
                  6_ground_truth_annotations.py reports as num_nodules)
    * luna16_gt: >=3 of 4 readers AND diameter >= 3mm - the definition the
                  detector was actually trained on (LUNA16), so this is the fairer
                  comparison for raw detector recall.
- A per-patient count table at one reference threshold (0.5, this pipeline's
  existing default) for a plain "how many did it find vs. ground truth" view.

Matching rule: a detection is a true positive for a ground-truth nodule if its
world-mm center falls within that nodule's radius (diameter_mm / 2) of the
nodule's centroid, matched greedily by descending detection score, one nodule per
detection and vice versa - this is the official LUNA16 challenge hit criterion.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from nodule_detector import DICOM_ROOT, build_detector, detect_patient

GT_PATH = Path(__file__).resolve().parent / "ground_truth_annotations.json"
DETECTION_CACHE_DIR = Path(__file__).resolve().parent / "detection_cache"
LOW_SCORE_THRESH = 0.02  # cast a wide net once per patient; sweep cutoffs in Python after
REFERENCE_THRESHOLD = 0.5  # this pipeline's existing default elsewhere (e.g. 5_characterize_nodules.py)
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)

LUNA16_MIN_ANNOTATIONS = 3
LUNA16_MIN_DIAMETER_MM = 3.0


def luna16_style(nodules):
    return [n for n in nodules if n["num_annotations"] >= LUNA16_MIN_ANNOTATIONS
                                and n["diameter_mm"] >= LUNA16_MIN_DIAMETER_MM]


def match(detections, gt_nodules):
    """Greedy nearest-centroid-within-radius matching, highest-score detection first.
    Returns (tp, fp, matched_gt_indices)."""
    remaining = set(range(len(gt_nodules)))
    tp = 0
    fp = 0
    for center, _score in sorted(detections, key=lambda d: -d[1]):
        best_idx, best_dist = None, np.inf
        for idx in remaining:
            dist = np.linalg.norm(np.array(center) - np.array(gt_nodules[idx]["centroid_mm"]))
            if dist < best_dist:
                best_idx, best_dist = idx, dist
        radius = gt_nodules[best_idx]["diameter_mm"] / 2 if best_idx is not None else -1
        if best_idx is not None and best_dist <= radius:
            tp += 1
            remaining.discard(best_idx)
        else:
            fp += 1
    return tp, fp, set(range(len(gt_nodules))) - remaining


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu",
                     help="Device for the detector's network forward pass; box selection/NMS "
                          "and sliding-window aggregation always run on CPU regardless (see "
                          "monai_bundles/lung_nodule_ct_detection/configs/inference.json).")
    args = ap.parse_args()

    patient_ids = [f"LIDC-IDRI-{i:04d}" for i in range(args.start, args.end + 1)]
    ground_truth = json.loads(GT_PATH.read_text())
    DETECTION_CACHE_DIR.mkdir(exist_ok=True)

    # --- Pass 1: run inference once per patient at a low score threshold, keep every candidate.
    # Cached to disk per patient (same pattern as 5_characterize_nodules.py's ROI cache) so a
    # long multi-patient run can be split across several shorter invocations - this machine runs
    # under enough memory pressure that a single uninterrupted multi-hour run isn't reliable, so
    # each patient's result is persisted as soon as it's computed and skipped on the next run.
    per_patient_detections = {}
    needs_detection = []
    for patient_id in patient_ids:
        cache_path = DETECTION_CACHE_DIR / f"{patient_id}.json"
        if cache_path.exists():
            per_patient_detections[patient_id] = json.loads(cache_path.read_text())
        elif ground_truth.get(patient_id, {}).get("found"):
            needs_detection.append(patient_id)

    if needs_detection:
        print(f"Loading detector on {args.device}...")
        detector, preprocessing, postprocessing = build_detector(args.device)

        for n, patient_id in enumerate(patient_ids):
            if patient_id in per_patient_detections:
                print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: already cached, skipping")
                continue
            if not ground_truth.get(patient_id, {}).get("found"):
                print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: no ground truth, skipping")
                continue
            try:
                _, world_boxes, scores, _, _ = detect_patient(
                    DICOM_ROOT / patient_id, detector, preprocessing, postprocessing, args.device, LOW_SCORE_THRESH
                )
                centers = world_boxes[:, :3].tolist()
                detections = list(zip(centers, scores.tolist()))
                (DETECTION_CACHE_DIR / f"{patient_id}.json").write_text(json.dumps(detections))
                per_patient_detections[patient_id] = detections
                print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: {len(centers)} candidate(s) "
                      f">= score {LOW_SCORE_THRESH}")
            except Exception as e:
                print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: FAILED - {type(e).__name__}: {e}")

        del detector, preprocessing, postprocessing
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    else:
        print("All requested patients already cached - skipping the detector entirely for this run.")

    # --- Pass 2: sweep score cutoffs against both ground-truth consensus definitions ---
    gt_by_patient = {
        pid: {
            "all_gt": ground_truth[pid]["nodules"],
            "luna16_gt": luna16_style(ground_truth[pid]["nodules"]),
        }
        for pid in per_patient_detections
    }

    froc_rows = []
    for cutoff in THRESHOLD_GRID:
        totals = {"all_gt": [0, 0, 0], "luna16_gt": [0, 0, 0]}  # tp, fp, n_gt
        for patient_id, detections in per_patient_detections.items():
            filtered = [(c, s) for c, s in detections if s >= cutoff]
            for key in ("all_gt", "luna16_gt"):
                gt_nodules = gt_by_patient[patient_id][key]
                tp, fp, _ = match(filtered, gt_nodules)
                totals[key][0] += tp
                totals[key][1] += fp
                totals[key][2] += len(gt_nodules)

        n_patients = len(per_patient_detections)
        row = {"score_threshold": float(cutoff)}
        for key in ("all_gt", "luna16_gt"):
            tp, fp, n_gt = totals[key]
            row[f"sensitivity_{key}"] = tp / n_gt if n_gt else 0.0
            row[f"avg_fp_per_scan_{key}"] = fp / n_patients
        froc_rows.append(row)

    froc_path = Path(__file__).resolve().parent / "detection_froc.csv"
    with froc_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=froc_rows[0].keys())
        writer.writeheader()
        writer.writerows(froc_rows)
    print(f"\nWrote FROC sweep to {froc_path}")

    # --- Reference-threshold per-patient count table ---
    print(f"\nPer-patient counts at score >= {REFERENCE_THRESHOLD}:")
    print(f"{'patient':<18}{'detected':>10}{'gt_all':>10}{'gt_luna16':>12}")
    count_rows = []
    for patient_id, detections in per_patient_detections.items():
        filtered = [(c, s) for c, s in detections if s >= REFERENCE_THRESHOLD]
        n_all = len(gt_by_patient[patient_id]["all_gt"])
        n_luna16 = len(gt_by_patient[patient_id]["luna16_gt"])
        print(f"{patient_id:<18}{len(filtered):>10}{n_all:>10}{n_luna16:>12}")
        count_rows.append({
            "patient_id": patient_id,
            "detected_count": len(filtered),
            "gt_count_all": n_all,
            "gt_count_luna16": n_luna16,
        })

    counts_path = Path(__file__).resolve().parent / "detection_counts.json"
    counts_path.write_text(json.dumps(count_rows, indent=2))
    print(f"\nWrote per-patient counts to {counts_path}")
