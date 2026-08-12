from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from transformers import pipeline

from medgemma_ct import hu_to_rgb, sample_slices

MODEL_ID = "google/medgemma-1.5-4b-it"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

DICOM_ROOT = Path(__file__).resolve().parent.parent / "datasets/LDIC-IDRI-subset/lidc_idri"
RESULTS_CSV = Path(__file__).resolve().parent / "nodule_counts_3d.csv"

PROMPT = '''You are a radiologist. You are given the full sequence of axial slices from a chest CT scan, ordered from superior to inferior and labeled with their slice number.
Examine the whole volume and determine the number of distinct lung nodules present, counting each nodule once even if it spans multiple slices. Nodules are small, round or oval-shaped growths in the lung parenchyma.
Respond with only the total nodule count as an integer. If there are none, respond with "0".'''


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
    """Load one DICOM slice and map it to the 3-channel HU window MedGemma 1.5 expects."""
    ds = pydicom.dcmread(dcm_path)
    slope = float(ds.get("RescaleSlope", 1))
    intercept = float(ds.get("RescaleIntercept", 0))
    hu = ds.pixel_array.astype(np.float32) * slope + intercept
    return hu_to_rgb(hu)


def build_messages(slice_paths):
    content = []
    for i, path in enumerate(slice_paths):
        content.append({"type": "text", "text": f"Slice {i + 1}/{len(slice_paths)}:"})
        content.append({"type": "image", "image": load_slice_rgb(path)})
    content.append({"type": "text", "text": PROMPT})
    return [{"role": "user", "content": content}]


pipe = pipeline("image-text-to-text", model=MODEL_ID, device=DEVICE, dtype=torch.bfloat16)

patient_dirs = sorted(d for d in DICOM_ROOT.iterdir() if d.is_dir())
print(f"Found {len(patient_dirs)} patients")

rows = []
for i, patient_dir in enumerate(patient_dirs):
    for series_uid, slice_paths in dicom_series_for_patient(patient_dir).items():
        sampled = sample_slices(slice_paths)
        messages = build_messages(sampled)
        result = pipe(text=messages)
        response = result[0]["generated_text"][-1]["content"]
        rows.append({
            "patient_id": patient_dir.name,
            "series_uid": series_uid,
            "num_slices_total": len(slice_paths),
            "num_slices_sampled": len(sampled),
            "response": response,
        })
        print(f"[{i + 1}/{len(patient_dirs)}] {patient_dir.name} ({series_uid[:8]}...): {response}")

results = pd.DataFrame(rows)
results.to_csv(RESULTS_CSV, index=False)
print(f"Saved results to {RESULTS_CSV}")
