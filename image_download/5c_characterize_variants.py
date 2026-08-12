"""A/B variants of ground-truth-location characterization (5b_characterize_ground_truth_nodules.py),
testing two of the cheap levers before reaching for LoRA fine-tuning:

  --anchored   Appends targeted anti-underrating guidance to the subtlety/margin/spiculation
               attribute descriptions - the three attributes 5_characterize_nodules.py's own
               comments (and bias_correction.json's measured -0.53/-0.55/-0.44 biases) flag as
               systematically under-rated. Free (no extra tokens beyond a few sentences).

  --fewshot    Prepends two fixed calibration examples (real images + their consensus-rounded
               ground-truth JSON) as prior conversation turns before the target nodule: one
               unambiguous low-spiculation nodule (LIDC-IDRI-0005 #0, spiculation=1.0/4 readers)
               and one unambiguous high-spiculation nodule (LIDC-IDRI-0007 #0, spiculation=5.0/4
               readers) - anchoring both ends of the scale the model was seen defaulting to "1"
               on. Costs ~2x the vision tokens/prefill time of the baseline per nodule.

Combinable (--anchored --fewshot), and each flag combination writes to its own output directory
so results never clobber the zero-shot baseline in nodule_characteristics_gt/ - use
compare_variants() (also exposed to the Colab notebook) to see MAE/bias side by side per variant.

Cached/resumable exactly like 5b: per-nodule, written to disk after every single call.
"""
import argparse
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
BASE_DIR = Path(__file__).resolve().parent

CHARACTERIZE_MAX_SLICES = 15
PAD_MM = 20.0
ATTRS = list(LIDC_ATTRIBUTES.keys())

# The two calibration examples for --fewshot: (patient_id, nodule_index), chosen from
# ground_truth_annotations.json for unambiguous, high-confidence (4/4 readers) spiculation at
# each extreme - see the conversation that picked these for the full candidate search.
FEWSHOT_EXAMPLES = [
    ("LIDC-IDRI-0005", 0),  # spiculation=1.00 (4/4 readers) - "No Spiculation" anchor
    ("LIDC-IDRI-0007", 0),  # spiculation=5.00 (4/4 readers) - "Marked Spiculation" anchor
]

# Extra guidance appended (only under --anchored) to the three attributes bias_correction.json
# measured as systematically under-rated: subtlety -0.53, margin -0.55, spiculation -0.44 (mean
# predicted-minus-ground-truth over 56 matched real nodules on the detector-based pipeline).
ANCHORED_ADDENDA = {
    "subtlety": (
        "Prior runs of this exact task showed a tendency to under-rate subtlety - if the nodule "
        "is clearly denser than the surrounding lung and easy to spot without hunting for it, "
        "score it 4-5. Do not default to a middling score out of caution."
    ),
    "margin": (
        "Prior runs of this exact task showed a tendency to under-rate margin definition - if the "
        "boundary is cleanly traceable across most of the slices where the nodule appears, score "
        "it 4-5 even if a few slices show some haziness. Do not default to a middling score out of "
        "caution."
    ),
    "spiculation": (
        "Prior runs of this exact task showed a strong tendency to default to 'No Spiculation' "
        "(score 1) even on markedly spiculated nodules. Check EVERY slice individually for thin "
        "strands radiating from the nodule into the surrounding lung, not just the overall "
        "silhouette - radiating lines visible across several slices should score 3 or higher, and "
        "a dense corona of strands should score 4-5. Only score 1 if the boundary is completely "
        "smooth with no radiating strands in any slice."
    ),
}

PROMPT_TEMPLATE = '''You are an expert radiologist. You are given a sequence of axial CT slices, ordered from superior to inferior, showing a small region of the lung centered on a known lung nodule.

Assess the nodule on each of the following 9 characteristics, using the exact integer scale given for each:

{attribute_guide}

Respond with ONLY a single JSON object with exactly these 9 keys: subtlety, internalStructure, calcification, sphericity, margin, lobulation, spiculation, texture, malignancy (each mapped to an integer from its scale above). Do not include any other text before or after the JSON.'''


def format_attribute_guide(anchored):
    lines = []
    for name, spec in LIDC_ATTRIBUTES.items():
        labels = ", ".join(f"{k}={v}" for k, v in spec["labels"].items())
        guidance = spec["guidance"]
        if anchored and name in ANCHORED_ADDENDA:
            guidance = f"{guidance} {ANCHORED_ADDENDA[name]}"
        lines.append(f'- "{name}" ({spec["scale"]}): {spec["description"]} {guidance} Values: {labels}.')
    return "\n".join(lines)


# --- ROI loading for the few-shot calibration examples (same DICOM loading/cropping as
# 5b_characterize_ground_truth_nodules.py, shared via ground_truth_roi.py) ---

def load_roi_for_nodule(patient_id, nodule_index, ground_truth, volume_cache=None):
    """Crop the ROI for one nodule, reusing `volume_cache[patient_id]` (a
    load_patient_volume() result) if already present, and populating it otherwise -
    lets the few-shot calibration patients' volumes be shared with the main loop
    instead of being read off disk twice when a calibration patient also falls
    inside --start/--end."""
    gt = ground_truth[patient_id]
    nodule = next(nd for nd in gt["nodules"] if nd["nodule_index"] == nodule_index)
    if volume_cache is not None and patient_id in volume_cache:
        volume, pixel_spacing, origin_xy, z_positions = volume_cache[patient_id]
    else:
        volume, pixel_spacing, origin_xy, z_positions = load_patient_volume(DICOM_ROOT / patient_id, gt["series_instance_uid"])
        if volume_cache is not None:
            volume_cache[patient_id] = (volume, pixel_spacing, origin_xy, z_positions)
    roi_hu = crop_at_centroid(volume, pixel_spacing, origin_xy, z_positions,
                               nodule["centroid_mm"], nodule["diameter_mm"], pad_mm=PAD_MM)
    means = {a: sum(ann[a] for ann in nodule["annotations"]) / len(nodule["annotations"]) for a in ATTRS}
    # Round-half-up, not Python's round-half-to-even: these means are always positive (1-5 scale
    # attributes), so int(x + 0.5) matches ordinary rounding without banker's-rounding pulling
    # exact .5 means down to the nearest even integer (e.g. 2.5 -> 2 instead of 3).
    rounded = {a: int(means[a] + 0.5) for a in ATTRS}
    return roi_hu, rounded


# --- Message building ---

def build_image_content(roi_hu):
    slices = sample_slices([roi_hu[:, :, i] for i in range(roi_hu.shape[2])], max_slices=CHARACTERIZE_MAX_SLICES)
    content = []
    for i, sl in enumerate(slices):
        content.append({"type": "text", "text": f"Slice {i + 1}/{len(slices)}:"})
        content.append({"type": "image", "image": hu_to_rgb(sl)})
    return content


def build_messages(target_roi, anchored, fewshot_examples=None):
    """fewshot_examples, if given, is a list of (image_content, answer_json) pairs with
    image_content already built by build_image_content() - the two calibration ROIs never
    change within a run, so the caller builds their image content once and passes it in here
    rather than this function re-encoding the same two images on every nodule."""
    attribute_guide = format_attribute_guide(anchored)
    instructions = PROMPT_TEMPLATE.format(attribute_guide=attribute_guide)

    if not fewshot_examples:
        content = build_image_content(target_roi) + [{"type": "text", "text": instructions}]
        return [{"role": "user", "content": content}]

    messages = []
    for i, (image_content, answer_json) in enumerate(fewshot_examples):
        content = list(image_content)  # copy - we append a per-call text block below
        content.append({"type": "text", "text": instructions if i == 0 else
                         "Here is another example nodule. Respond with ONLY the JSON object, exactly as before."})
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": json.dumps(answer_json)}]})

    content = build_image_content(target_roi)
    content.append({"type": "text", "text":
                     "Now assess this new nodule. Respond with ONLY the JSON object, exactly as in the examples above."})
    messages.append({"role": "user", "content": content})
    return messages


def parse_json_response(response):
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_attributes(attributes):
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


def variant_dir(anchored, fewshot):
    suffix = ("_anchored" if anchored else "") + ("_fewshot" if fewshot else "")
    return BASE_DIR / f"nodule_characteristics_gt{suffix}"


def remaining_nodules(patient_id, ground_truth, output_dir, fewshot):
    gt = ground_truth.get(patient_id, {})
    if not gt.get("found"):
        return [], gt
    out_path = output_dir / f"{patient_id}.json"
    done = set()
    if out_path.exists():
        done = {r["nodule_index"] for r in json.loads(out_path.read_text())}
    # Only exclude the few-shot calibration nodules from scoring when this run actually shows
    # the model their answer in-context (--fewshot) - an --anchored-only run never sees them as
    # examples, so there's no leakage risk and excluding them would just cost 2 nodules' worth of
    # data for nothing, making it a smaller (and non-apples-to-apples) sample than the baseline.
    fewshot_keys = {(pid, idx) for pid, idx in FEWSHOT_EXAMPLES} if fewshot else set()
    return [nd for nd in gt["nodules"]
            if nd["nodule_index"] not in done and (patient_id, nd["nodule_index"]) not in fewshot_keys], gt


def compare_variants(variant_dirs):
    """Print MAE/bias per attribute side by side for several output directories - pass a
    {label: Path} dict. Used by both this script's own summary and the notebook's compare cell."""
    all_data = {}
    for label, d in variant_dirs.items():
        rows = []
        for f in sorted(Path(d).glob("LIDC-IDRI-*.json")):
            rows.extend(json.loads(f.read_text()))
        all_data[label] = rows

    header = f"{'attribute':<18}" + "".join(f"{label + ' MAE':>16}{label + ' bias':>16}" for label in all_data)
    print(header)
    for a in ATTRS:
        line = f"{a:<18}"
        for label, rows in all_data.items():
            errs = [r[f"{a}_err"] for r in rows if r.get(f"{a}_err") is not None]
            mae = sum(abs(e) for e in errs) / len(errs) if errs else None
            bias = sum(errs) / len(errs) if errs else None
            line += f"{(f'{mae:.2f}' if mae is not None else 'n/a'):>16}"
            line += f"{(f'{bias:+.2f}' if bias is not None else 'n/a'):>16}"
        print(line)
    return all_data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--anchored", action="store_true", help="Use anti-underrating guidance for subtlety/margin/spiculation.")
    ap.add_argument("--fewshot", action="store_true", help="Prepend the two fixed calibration examples.")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    patient_ids = [f"LIDC-IDRI-{i:04d}" for i in range(args.start, args.end + 1)]
    output_dir = variant_dir(args.anchored, args.fewshot)
    output_dir.mkdir(exist_ok=True)
    print(f"Variant: anchored={args.anchored} fewshot={args.fewshot} -> {output_dir}")

    ground_truth = json.loads(GT_PATH.read_text())

    per_patient_remaining = {pid: remaining_nodules(pid, ground_truth, output_dir, args.fewshot)[0] for pid in patient_ids}
    total_remaining = sum(len(v) for v in per_patient_remaining.values())
    needs_run = [pid for pid, nds in per_patient_remaining.items() if nds]

    pipe = None
    fewshot_examples = None
    # Populated with any FEWSHOT_EXAMPLES patient's (volume, pixel_spacing, origin_xy,
    # z_positions) below, so the main per-patient loop reuses it instead of re-parsing that
    # patient's whole DICOM series from disk if it also falls inside --start/--end.
    volume_cache = {}
    if needs_run:
        print(f"Loading MedGemma 1.5 on {DEVICE}... ({total_remaining} nodule(s) to characterize)")
        pipe = pipeline("image-text-to-text", model=MODEL_ID, device=DEVICE, dtype=torch.bfloat16)
        if args.fewshot:
            print("Cropping few-shot calibration examples...")
            fewshot_examples = [
                (build_image_content(roi_hu), answer_json)
                for roi_hu, answer_json in
                (load_roi_for_nodule(pid, idx, ground_truth, volume_cache) for pid, idx in FEWSHOT_EXAMPLES)
            ]
    else:
        print("All requested patients' nodules already characterized - skipping MedGemma entirely for this run.")

    nodule_seconds = []
    nodules_done = 0

    for n, patient_id in enumerate(patient_ids):
        todo, gt = remaining_nodules(patient_id, ground_truth, output_dir, args.fewshot)
        if not gt.get("found"):
            print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: no ground truth, skipping")
            continue
        if not todo:
            print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: already characterized, skipping")
            continue

        out_path = output_dir / f"{patient_id}.json"
        patient_rows = json.loads(out_path.read_text()) if out_path.exists() else []

        series_uid = gt["series_instance_uid"]
        if patient_id in volume_cache:
            volume, pixel_spacing, origin_xy, z_positions = volume_cache.pop(patient_id)
        else:
            volume, pixel_spacing, origin_xy, z_positions = load_patient_volume(DICOM_ROOT / patient_id, series_uid)

        for nodule in todo:
            roi_hu = crop_at_centroid(
                volume, pixel_spacing, origin_xy, z_positions,
                nodule["centroid_mm"], nodule["diameter_mm"], pad_mm=PAD_MM,
            )
            messages = build_messages(roi_hu, args.anchored, fewshot_examples)
            response, elapsed = run_pipe(pipe, messages, args.max_new_tokens)
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
            out_path.write_text(json.dumps(patient_rows, indent=2))

            remaining = total_remaining - nodules_done
            eta_note = ""
            if nodule_seconds and remaining > 0:
                avg = sum(nodule_seconds) / len(nodule_seconds)
                eta_note = f"  ETA {format_duration(avg * remaining)} ({remaining} nodule(s) left)"
            status = "OK" if not problems else f"PROBLEMS: {problems}"
            print(f"[{n + 1}/{len(patient_ids)}] {patient_id} nodule {nodule['nodule_index']} "
                  f"({elapsed:.1f}s): {status}{eta_note}")

        print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: {len(patient_rows)} nodule(s) -> {out_path}")

    if pipe is not None:
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    print(f"\n--- Summary for this variant ({output_dir.name}) ---")
    compare_variants({output_dir.name: output_dir})
