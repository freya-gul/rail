"""Count lung nodules with MedGemma alone (whole-volume, single global count per
scan) and compare against two independent ground truths - a MedGemma-only
counting baseline to set against the MONAI detector evaluated in
7_evaluate_detection.py.

Cached per patient (same pattern as 5_characterize_nodules.py /
7_evaluate_detection.py) so a long run can be resumed: for each patient, run
MedGemma 1.5 on the full sampled slice sequence of each CT series (same slice
sampling/windowing as 2_model_3d.py), parse an integer nodule count from its
response, cache the raw response + parsed count to
medgemma_count_cache/<patient>.json, and immediately print that patient's
count against two genuinely separate ground truths (plus a running ETA):
  * gt_count_pylidc: ground_truth_annotations.json's num_nodules - every
                      pylidc consensus annotation cluster, any number of
                      readers.
  * gt_count_luna16: the actual external LUNA16 challenge annotations.csv
                      (not a rule applied to pylidc's own data - a
                      separately collected/filtered nodule list), joined to
                      patients via SeriesInstanceUID.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import pydicom
import torch
from transformers import pipeline

from medgemma_ct import hu_to_rgb, sample_slices

MODEL_ID = "google/medgemma-1.5-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

DICOM_ROOT = Path(__file__).resolve().parent.parent / "datasets/LDIC-IDRI-subset/lidc_idri"
GT_PATH = Path(__file__).resolve().parent / "ground_truth_annotations.json"
LUNA16_ANNOTATIONS_CSV = Path(__file__).resolve().parent.parent / "datasets/LDIC-IDRI-subset/annotations.csv"
CACHE_DIR = Path(__file__).resolve().parent / "medgemma_count_cache"
OUTPUT_CSV = Path(__file__).resolve().parent / "medgemma_counting_comparison.csv"

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


def luna16_csv_counts_by_series():
    """Nodule count per SeriesInstanceUID straight from the external LUNA16
    challenge annotations.csv - one row per annotated nodule, so grouping by
    seriesuid gives LUNA16's own count, independent of pylidc's clustering."""
    counts = {}
    with LUNA16_ANNOTATIONS_CSV.open() as f:
        for row in csv.DictReader(f):
            counts[row["seriesuid"]] = counts.get(row["seriesuid"], 0) + 1
    return counts


def format_duration(seconds):
    seconds = round(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    patient_ids = [f"LIDC-IDRI-{i:04d}" for i in range(args.start, args.end + 1)]
    CACHE_DIR.mkdir(exist_ok=True)

    ground_truth = json.loads(GT_PATH.read_text())
    luna16_csv_counts = luna16_csv_counts_by_series()

    needs_run = [pid for pid in patient_ids if not (CACHE_DIR / f"{pid}.json").exists()]
    pipe = None
    if needs_run:
        print(f"Loading MedGemma 1.5 on {DEVICE}...")
        pipe = pipeline("image-text-to-text", model=MODEL_ID, device=DEVICE, dtype=torch.bfloat16)
    else:
        print("All requested patients already cached - skipping MedGemma entirely for this run.")

    # Seconds spent actually running MedGemma per patient (cache hits don't count - they're
    # ~instant and would understate the ETA for the patients still to come).
    patient_seconds = []

    rows = []
    for n, patient_id in enumerate(patient_ids):
        cache_path = CACHE_DIR / f"{patient_id}.json"
        if cache_path.exists():
            series_results = json.loads(cache_path.read_text())
            timing_note = "cached"
        else:
            patient_dir = DICOM_ROOT / patient_id
            if not patient_dir.is_dir():
                print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: no DICOM directory, skipping")
                continue

            patient_start = time.monotonic()
            series = dicom_series_for_patient(patient_dir)
            series_results = []
            for series_uid, slice_paths in series.items():
                sampled = sample_slices(slice_paths)
                messages = build_messages(sampled)
                result = pipe(text=messages, max_new_tokens=args.max_new_tokens)
                response = result[0]["generated_text"][-1]["content"]
                count = parse_count(response)
                series_results.append({
                    "series_uid": series_uid,
                    "num_slices_total": len(slice_paths),
                    "num_slices_sampled": len(sampled),
                    "response": response,
                    "predicted_count": count,
                })
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            cache_path.write_text(json.dumps(series_results, indent=2))

            elapsed = time.monotonic() - patient_start
            patient_seconds.append(elapsed)
            timing_note = format_duration(elapsed)

        # A patient can have >1 CT series; sum predicted counts across series to
        # match ground truth, which is reported per patient in this pipeline.
        predicted = sum(r["predicted_count"] for r in series_results if r["predicted_count"] is not None)
        unparsed = sum(1 for r in series_results if r["predicted_count"] is None)

        gt = ground_truth.get(patient_id, {})
        gt_pylidc = len(gt.get("nodules", []))
        series_uid = gt.get("series_instance_uid")
        gt_luna16 = luna16_csv_counts.get(series_uid, 0) if series_uid else 0

        row = {
            "patient_id": patient_id,
            "predicted_count": predicted,
            "gt_count_pylidc": gt_pylidc,
            "gt_count_luna16": gt_luna16,
            "error_vs_pylidc": predicted - gt_pylidc,
            "error_vs_luna16": predicted - gt_luna16,
            "unparsed_responses": unparsed,
        }
        rows.append(row)

        eta_note = ""
        remaining = len(needs_run) - len(patient_seconds)
        if patient_seconds and remaining > 0:
            avg = sum(patient_seconds) / len(patient_seconds)
            eta_note = f"  ETA {format_duration(avg * remaining)} ({remaining} patient(s) left)"

        print(f"[{n + 1}/{len(patient_ids)}] {patient_id} ({timing_note}): "
              f"medgemma={predicted}  pylidc={gt_pylidc} (err {row['error_vs_pylidc']:+d})  "
              f"luna16={gt_luna16} (err {row['error_vs_luna16']:+d}){eta_note}")

    if pipe is not None:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"\n{'patient':<18}{'medgemma':>10}{'gt_pylidc':>11}{'gt_luna16':>11}{'err_pylidc':>12}{'err_luna16':>12}")
    for r in rows:
        print(f"{r['patient_id']:<18}{r['predicted_count']:>10}{r['gt_count_pylidc']:>11}"
              f"{r['gt_count_luna16']:>11}{r['error_vs_pylidc']:>12}{r['error_vs_luna16']:>12}")

    if rows:
        mae_pylidc = sum(abs(r["error_vs_pylidc"]) for r in rows) / len(rows)
        mae_luna16 = sum(abs(r["error_vs_luna16"]) for r in rows) / len(rows)
        bias_pylidc = sum(r["error_vs_pylidc"] for r in rows) / len(rows)
        bias_luna16 = sum(r["error_vs_luna16"] for r in rows) / len(rows)
        print(f"\nMAE vs pylidc: {mae_pylidc:.2f}   bias (pred-gt): {bias_pylidc:+.2f}")
        print(f"MAE vs luna16: {mae_luna16:.2f}   bias (pred-gt): {bias_luna16:+.2f}")

    with OUTPUT_CSV.open("w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "patient_id", "predicted_count", "gt_count_pylidc", "gt_count_luna16",
            "error_vs_pylidc", "error_vs_luna16", "unparsed_responses",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote comparison to {OUTPUT_CSV}")
