"""Count lung nodules with MedGemma alone (whole-volume, single global count per
scan) and compare against pylidc ground truth - a MedGemma-only counting
baseline to set against the MONAI detector evaluated in 7_evaluate_detection.py.

Two-pass, cached per patient (same pattern as 5_characterize_nodules.py /
7_evaluate_detection.py) so a long run can be resumed:
  Pass 1: run MedGemma 1.5 on the full sampled slice sequence of each patient's
          CT series (same slice sampling/windowing as 2_model_3d.py), parse an
          integer nodule count from its response, and cache the raw response +
          parsed count to medgemma_count_cache/<patient>.json.
  Pass 2: compare parsed counts against ground_truth_annotations.json's per-
          patient nodule count, against both ground-truth definitions used
          elsewhere in this pipeline:
            * all_gt:    every pylidc annotation cluster, any number of readers
            * luna16_gt: >=3 of 4 readers AND diameter >= 3mm (the LUNA16-style
                          definition used in 7_evaluate_detection.py)
"""
import argparse
import csv
import json
import re
from pathlib import Path

import pydicom
import torch
from transformers import pipeline

from medgemma_ct import hu_to_rgb, sample_slices

MODEL_ID = "google/medgemma-1.5-4b-it"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DICOM_ROOT = Path(__file__).resolve().parent.parent / "datasets/LDIC-IDRI-subset/lidc_idri"
GT_PATH = Path(__file__).resolve().parent / "ground_truth_annotations.json"
CACHE_DIR = Path(__file__).resolve().parent / "medgemma_count_cache"
OUTPUT_CSV = Path(__file__).resolve().parent / "medgemma_counting_comparison.csv"

LUNA16_MIN_ANNOTATIONS = 3
LUNA16_MIN_DIAMETER_MM = 3.0

PROMPT = '''You are a radiologist. You are given the full sequence of axial slices from a chest CT scan, ordered from superior to inferior and labeled with their slice number.
Examine the whole volume and determine the number of distinct lung nodules present, counting each nodule once even if it spans multiple slices. Nodules are small, round or oval-shaped growths in the lung parenchyma.

You may reason or write findings first if you want to, but no matter what else you write, the VERY LAST LINE of your response MUST be exactly of the form "COUNT: <integer>" (e.g. "COUNT: 3"), with nothing after it - including when your conclusion is that there are no nodules, in which case the last line must be "COUNT: 0". Never end your response without this line.'''


def dicom_series_for_patient(patient_dir):
    """Group a patient's CT DICOM files by SeriesInstanceUID, sorted by InstanceNumber."""
    series = {}
    for f in patient_dir.rglob("*.dcm"):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        if ds.Modality != "CT":
            continue
        series.setdefault(ds.SeriesInstanceUID, []).append((int(ds.InstanceNumber), f))
    return {uid: [p for _, p in sorted(slices)] for uid, slices in series.items()}


def load_slice_rgb(dcm_path):
    ds = pydicom.dcmread(dcm_path)
    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    hu = ds.pixel_array.astype("float32") * slope + intercept
    return hu_to_rgb(hu)


def build_messages(slice_paths):
    content = []
    for i, path in enumerate(slice_paths):
        content.append({"type": "text", "text": f"Slice {i + 1}/{len(slice_paths)}:"})
        content.append({"type": "image", "image": load_slice_rgb(path)})
    content.append({"type": "text", "text": PROMPT})
    return [{"role": "user", "content": content}]


def parse_count(response):
    """Pull the integer out of a 'COUNT: N' line; fall back to the last standalone
    integer in the response if MedGemma didn't follow the format exactly (it
    sometimes reasons in prose first - see 5_characterize_nodules.py's note on
    MedGemma 1.5's visible 'thought' block)."""
    match = re.search(r"COUNT:\s*(\d+)", response, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    numbers = re.findall(r"\b\d+\b", response)
    return int(numbers[-1]) if numbers else None


def luna16_style(nodules):
    return [n for n in nodules if n["num_annotations"] >= LUNA16_MIN_ANNOTATIONS
                                and n["diameter_mm"] >= LUNA16_MIN_DIAMETER_MM]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    patient_ids = [f"LIDC-IDRI-{i:04d}" for i in range(args.start, args.end + 1)]
    CACHE_DIR.mkdir(exist_ok=True)

    needs_run = [pid for pid in patient_ids if not (CACHE_DIR / f"{pid}.json").exists()]

    if needs_run:
        print(f"Loading MedGemma 1.5 on {DEVICE}...")
        pipe = pipeline("image-text-to-text", model=MODEL_ID, device=DEVICE, dtype=torch.bfloat16)

        for n, patient_id in enumerate(patient_ids):
            cache_path = CACHE_DIR / f"{patient_id}.json"
            if cache_path.exists():
                print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: already cached, skipping")
                continue
            patient_dir = DICOM_ROOT / patient_id
            if not patient_dir.is_dir():
                print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: no DICOM directory, skipping")
                continue

            series = dicom_series_for_patient(patient_dir)
            results = []
            for series_uid, slice_paths in series.items():
                sampled = sample_slices(slice_paths)
                messages = build_messages(sampled)
                result = pipe(text=messages, max_new_tokens=args.max_new_tokens)
                response = result[0]["generated_text"][-1]["content"]
                count = parse_count(response)
                results.append({
                    "series_uid": series_uid,
                    "num_slices_total": len(slice_paths),
                    "num_slices_sampled": len(sampled),
                    "response": response,
                    "predicted_count": count,
                })
                print(f"[{n + 1}/{len(patient_ids)}] {patient_id} ({series_uid[:8]}...): "
                      f"predicted_count={count}  raw={response!r}")
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            cache_path.write_text(json.dumps(results, indent=2))

        del pipe
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    else:
        print("All requested patients already cached - skipping MedGemma entirely for this run.")

    # --- Compare against ground truth ---
    ground_truth = json.loads(GT_PATH.read_text())
    rows = []
    for patient_id in patient_ids:
        cache_path = CACHE_DIR / f"{patient_id}.json"
        if not cache_path.exists():
            continue
        series_results = json.loads(cache_path.read_text())
        # A patient can have >1 CT series; sum predicted counts across series to
        # match ground truth, which is reported per patient in this pipeline.
        predicted = sum(r["predicted_count"] for r in series_results if r["predicted_count"] is not None)
        unparsed = sum(1 for r in series_results if r["predicted_count"] is None)

        gt = ground_truth.get(patient_id, {})
        gt_nodules = gt.get("nodules", [])
        gt_all = len(gt_nodules)
        gt_luna16 = len(luna16_style(gt_nodules))

        rows.append({
            "patient_id": patient_id,
            "predicted_count": predicted,
            "gt_count_all": gt_all,
            "gt_count_luna16": gt_luna16,
            "error_vs_all": predicted - gt_all,
            "error_vs_luna16": predicted - gt_luna16,
            "unparsed_responses": unparsed,
        })

    print(f"\n{'patient':<18}{'medgemma':>10}{'gt_all':>10}{'gt_luna16':>12}{'err_all':>10}{'err_luna16':>12}")
    for r in rows:
        print(f"{r['patient_id']:<18}{r['predicted_count']:>10}{r['gt_count_all']:>10}"
              f"{r['gt_count_luna16']:>12}{r['error_vs_all']:>10}{r['error_vs_luna16']:>12}")

    if rows:
        mae_all = sum(abs(r["error_vs_all"]) for r in rows) / len(rows)
        mae_luna16 = sum(abs(r["error_vs_luna16"]) for r in rows) / len(rows)
        bias_all = sum(r["error_vs_all"] for r in rows) / len(rows)
        bias_luna16 = sum(r["error_vs_luna16"] for r in rows) / len(rows)
        print(f"\nMAE vs all_gt: {mae_all:.2f}   bias (pred-gt): {bias_all:+.2f}")
        print(f"MAE vs luna16_gt: {mae_luna16:.2f}   bias (pred-gt): {bias_luna16:+.2f}")

    with OUTPUT_CSV.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "patient_id", "predicted_count", "gt_count_all", "gt_count_luna16",
            "error_vs_all", "error_vs_luna16", "unparsed_responses",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote comparison to {OUTPUT_CSV}")
