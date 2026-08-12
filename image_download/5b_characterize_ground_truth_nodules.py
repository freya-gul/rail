"""Characterize nodules at their pylidc GROUND-TRUTH locations with MedGemma,
skipping the MONAI detector entirely - a sibling to 5_characterize_nodules.py
that measures MedGemma's characterization accuracy on its own, without also
folding in detector recall/false-positive noise or a detection-to-ground-truth
matching step.

For each pylidc consensus nodule in ground_truth_annotations.json (already
committed for patients 1-20), crop a padded 3D region directly out of the
patient's own DICOM series around that nodule's own centroid_mm/diameter_mm
(no detector, no resampling grid), ask MedGemma to rate it on the same 9
pylidc attributes as 5_characterize_nodules.py, and compare directly against
the mean of that nodule's own radiologist annotations.

Cached per patient AND resumable mid-patient (a patient with N nodules that
gets interrupted after M<N of them will pick back up at nodule M+1 on rerun,
rather than redoing the whole patient): results accumulate in
nodule_characteristics_gt/<patient>.json.
"""
import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from transformers import pipeline

from ground_truth_roi import crop_at_centroid, load_patient_volume
from lidc_attributes import LIDC_ATTRIBUTES
from medgemma_ct import hu_to_rgb, sample_slices

MODEL_ID = "google/medgemma-1.5-4b-it"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

DICOM_ROOT = Path(__file__).resolve().parent.parent / "datasets/LDIC-IDRI-subset/lidc_idri"
GT_PATH = Path(__file__).resolve().parent / "ground_truth_annotations.json"
OUTPUT_DIR = Path(__file__).resolve().parent / "nodule_characteristics_gt"
SUMMARY_PATH = Path(__file__).resolve().parent / "characterize_ground_truth_summary.json"
CSV_PATH = Path(__file__).resolve().parent / "characterize_ground_truth_comparison.csv"

CHARACTERIZE_MAX_SLICES = 15
PAD_MM = 20.0  # margin around the nodule's own radius, in every axis - same idea as roi_utils.PAD_MM

ATTRS = list(LIDC_ATTRIBUTES.keys())

PROMPT_TEMPLATE = '''You are an expert radiologist. You are given a sequence of axial CT slices, ordered from superior to inferior, showing a small region of the lung centered on a known lung nodule.

Assess the nodule on each of the following 9 characteristics, using the exact integer scale given for each:

{attribute_guide}

Respond with ONLY a single JSON object with exactly these 9 keys: subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture, malignancy (each mapped to an integer from its scale above). Do not include any other text before or after the JSON.'''


def format_attribute_guide():
    lines = []
    for name, spec in LIDC_ATTRIBUTES.items():
        labels = ", ".join(f"{k}={v}" for k, v in spec["labels"].items())
        lines.append(f'- "{name}" ({spec["scale"]}): {spec["description"]} {spec["guidance"]} Values: {labels}.')
    return "\n".join(lines)


ATTRIBUTE_GUIDE = format_attribute_guide()  # built once - identical for every nodule/patient


def build_messages(roi_hu):
    slices = sample_slices([roi_hu[:, :, i] for i in range(roi_hu.shape[2])], max_slices=CHARACTERIZE_MAX_SLICES)
    content = []
    for i, sl in enumerate(slices):
        content.append({"type": "text", "text": f"Slice {i + 1}/{len(slices)}:"})
        content.append({"type": "image", "image": hu_to_rgb(sl)})
    content.append({"type": "text", "text": PROMPT_TEMPLATE.format(attribute_guide=ATTRIBUTE_GUIDE)})
    return [{"role": "user", "content": content}]


def parse_json_response(response):
    """Extract the first {...} JSON object from a model response, tolerating stray prose/fences."""
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_attributes(attributes):
    """Return a list of problems (missing keys, out-of-range values) - empty list if all good."""
    if attributes is None:
        return ["response was not valid JSON"]
    problems = []
    for name, spec in LIDC_ATTRIBUTES.items():
        if name not in attributes:
            problems.append(f"missing key '{name}'")
        elif attributes[name] not in spec["labels"]:
            problems.append(f"'{name}'={attributes[name]!r} not in valid set {sorted(spec['labels'])}")
    return problems


def run_pipe(pipe, messages, max_new_tokens=1024):
    # MedGemma 1.5 emits a visible reasoning ("thought") block before its actual answer - give it
    # enough budget to finish reasoning AND answer (see 5_characterize_nodules.py's note on this).
    start = time.monotonic()
    result = pipe(text=messages, max_new_tokens=max_new_tokens)
    elapsed = time.monotonic() - start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result[0]["generated_text"][-1]["content"], elapsed


def format_duration(seconds):
    seconds = round(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s" if m else f"{s}s"


def gt_means(nodule):
    return {a: sum(ann[a] for ann in nodule["annotations"]) / len(nodule["annotations"]) for a in ATTRS}


def remaining_nodules(patient_id, ground_truth):
    """Nodules not yet in this patient's output file - supports resuming mid-patient."""
    gt = ground_truth.get(patient_id, {})
    if not gt.get("found"):
        return [], gt
    out_path = OUTPUT_DIR / f"{patient_id}.json"
    done = set()
    if out_path.exists():
        done = {r["nodule_index"] for r in json.loads(out_path.read_text())}
    return [nd for nd in gt["nodules"] if nd["nodule_index"] not in done], gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    patient_ids = [f"LIDC-IDRI-{i:04d}" for i in range(args.start, args.end + 1)]
    OUTPUT_DIR.mkdir(exist_ok=True)

    ground_truth = json.loads(GT_PATH.read_text())

    per_patient_remaining = {pid: remaining_nodules(pid, ground_truth)[0] for pid in patient_ids}
    total_remaining = sum(len(v) for v in per_patient_remaining.values())
    needs_run = [pid for pid, nds in per_patient_remaining.items() if nds]

    pipe = None
    if needs_run:
        print(f"Loading MedGemma 1.5 on {DEVICE}... ({total_remaining} nodule(s) to characterize)")
        pipe = pipeline("image-text-to-text", model=MODEL_ID, device=DEVICE, dtype=torch.bfloat16)
    else:
        print("All requested patients' nodules already characterized - skipping MedGemma entirely for this run.")

    nodule_seconds = []
    nodules_done = 0

    for n, patient_id in enumerate(patient_ids):
        todo, gt = remaining_nodules(patient_id, ground_truth)
        if not gt.get("found"):
            print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: no ground truth, skipping")
            continue
        if not todo:
            print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: already characterized "
                  f"({len(gt['nodules'])} nodule(s)), skipping")
            continue

        out_path = OUTPUT_DIR / f"{patient_id}.json"
        patient_rows = json.loads(out_path.read_text()) if out_path.exists() else []

        series_uid = gt["series_instance_uid"]
        volume, pixel_spacing, origin_xy, z_positions = load_patient_volume(DICOM_ROOT / patient_id, series_uid)

        for nodule in todo:
            roi_hu = crop_at_centroid(
                volume, pixel_spacing, origin_xy, z_positions,
                nodule["centroid_mm"], nodule["diameter_mm"], pad_mm=PAD_MM,
            )
            response, elapsed = run_pipe(pipe, build_messages(roi_hu))
            nodule_seconds.append(elapsed)
            nodules_done += 1

            predicted = parse_json_response(response)
            problems = validate_attributes(predicted)
            means = gt_means(nodule)

            row = {
                "patient_id": patient_id,
                "nodule_index": nodule["nodule_index"],
                "num_annotations": nodule["num_annotations"],
                "diameter_mm": round(nodule["diameter_mm"], 2),
                "validation_problems": problems,
                "raw_response": response,
                "characterize_seconds": round(elapsed, 1),
            }
            for a in ATTRS:
                pred_val = predicted.get(a) if predicted else None
                row[f"{a}_pred"] = pred_val
                row[f"{a}_gt"] = round(means[a], 2)
                row[f"{a}_err"] = round(pred_val - means[a], 2) if isinstance(pred_val, (int, float)) else None

            patient_rows.append(row)
            out_path.write_text(json.dumps(patient_rows, indent=2))  # write after every nodule, not just per-patient

            remaining = total_remaining - nodules_done
            eta_note = ""
            if nodule_seconds and remaining > 0:
                avg = sum(nodule_seconds) / len(nodule_seconds)
                eta_note = f"  ETA {format_duration(avg * remaining)} ({remaining} nodule(s) left)"
            status = "OK" if not problems else f"PROBLEMS: {problems}"
            print(f"[{n + 1}/{len(patient_ids)}] {patient_id} nodule {nodule['nodule_index']} "
                  f"({elapsed:.1f}s, {nodule['num_annotations']} reader(s)): {status}{eta_note}")

        print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: {len(patient_rows)} nodule(s) -> {out_path}")

    if pipe is not None:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # --- Gather every patient's output (including ones skipped this run) and summarize ---
    all_rows = []
    for patient_id in patient_ids:
        out_path = OUTPUT_DIR / f"{patient_id}.json"
        if out_path.exists():
            all_rows.extend(json.loads(out_path.read_text()))

    print(f"\n{'attribute':<18}{'MAE':>8}{'bias':>8}{'n':>6}   scale")
    summary = {}
    for a in ATTRS:
        errs = [r[f"{a}_err"] for r in all_rows if r[f"{a}_err"] is not None]
        mae = sum(abs(e) for e in errs) / len(errs) if errs else None
        bias = sum(errs) / len(errs) if errs else None
        lo, hi = min(LIDC_ATTRIBUTES[a]["labels"]), max(LIDC_ATTRIBUTES[a]["labels"])
        summary[a] = {"mae": mae, "bias": bias, "n": len(errs)}
        mae_str = f"{mae:.2f}" if mae is not None else "n/a"
        bias_str = f"{bias:+.2f}" if bias is not None else "n/a"
        print(f"{a:<18}{mae_str:>8}{bias_str:>8}{len(errs):>6}   {lo}-{hi}")

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote per-attribute summary to {SUMMARY_PATH}")

    if all_rows:
        fieldnames = ["patient_id", "nodule_index", "num_annotations", "diameter_mm"]
        for a in ATTRS:
            fieldnames += [f"{a}_pred", f"{a}_gt", f"{a}_err"]
        fieldnames += ["validation_problems", "characterize_seconds"]
        with CSV_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in all_rows:
                writer.writerow({**r, "validation_problems": "; ".join(r["validation_problems"])})
        print(f"Wrote flat comparison to {CSV_PATH}")
