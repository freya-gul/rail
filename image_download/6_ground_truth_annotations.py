import argparse
from pathlib import Path
import json

import numpy as np
import pydicom

# pylidc 0.2.3 (latest on PyPI) still calls the long-removed np.int/np.float/np.bool
# aliases (e.g. in Contour.to_matrix() and Annotation.surface_area/volume), which
# cluster_annotations() depends on - patch them back in rather than downgrade the
# shared venv's numpy, which other scripts/torch here rely on.
for _name, _builtin in [("int", int), ("float", float), ("bool", bool)]:
    if not hasattr(np, _name):
        setattr(np, _name, _builtin)

import pylidc as pl

from lidc_attributes import LIDC_ATTRIBUTES

OUTPUT_PATH = Path(__file__).resolve().parent / "ground_truth_annotations.json"
DICOM_ROOT = Path(__file__).resolve().parent.parent / "datasets/LDIC-IDRI-subset/lidc_idri"

CHARACTERISTIC_FIELDS = list(LIDC_ATTRIBUTES.keys())


def series_origin_xy(patient_id, series_uid):
    """(x0, y0) from DICOM ImagePositionPatient - constant across an axial series'
    slices, so any one file for this series gives the in-plane world origin. Reads
    directly via our own DICOM_ROOT rather than pylidc's own file-path resolution,
    which depends on a `.pylidcrc`-driven configparser.SafeConfigParser call that's
    been removed in Python 3.12+."""
    for f in (DICOM_ROOT / patient_id).rglob("*.dcm"):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        if ds.SeriesInstanceUID == series_uid:
            return float(ds.ImagePositionPatient[0]), float(ds.ImagePositionPatient[1])
    return None


def centroid_mm(ann, origin_xy, slice_zvals):
    """World-mm centroid for one annotation: pylidc's own `centroid` is a (i,j,k)
    voxel-index triple into `scan.to_volume()`, so convert in-plane via the scan's
    pixel spacing + DICOM origin, and out-of-plane by interpolating the fractional
    slice index against the scan's actual (sorted) slice z-positions."""
    i, j, k = ann.centroid
    x0, y0 = origin_xy
    ps = ann.scan.pixel_spacing
    x = x0 + j * ps
    y = y0 + i * ps
    z = float(np.interp(k, np.arange(len(slice_zvals)), slice_zvals))
    return [x, y, z]


def annotation_to_dict(ann, origin_xy, slice_zvals):
    data = {
        "annotation_id": ann.id,
        "diameter_mm": ann.diameter,
        "surface_area_mm2": ann.surface_area,
        "volume_mm3": ann.volume,
        "centroid_mm": centroid_mm(ann, origin_xy, slice_zvals),
    }
    for field in CHARACTERISTIC_FIELDS:
        data[field] = getattr(ann, field)
    return data


def collect_scan(scan):
    """Cluster a scan's per-radiologist annotations into distinct nodules (pylidc's
    standard way of turning up-to-4-reader markup into a single ground-truth nodule
    list), keeping each reader's own diameter/characteristics within the cluster."""
    clusters = scan.cluster_annotations()
    origin_xy = series_origin_xy(scan.patient_id, scan.series_instance_uid)
    slice_zvals = scan.slice_zvals

    nodules = []
    for i, cluster in enumerate(clusters):
        annotations = [annotation_to_dict(ann, origin_xy, slice_zvals) for ann in cluster]
        centroid = np.mean([a["centroid_mm"] for a in annotations], axis=0).tolist()
        nodules.append({
            "nodule_index": i,
            "num_annotations": len(annotations),
            "diameter_mm": sum(a["diameter_mm"] for a in annotations) / len(annotations),
            "centroid_mm": centroid,
            "annotations": annotations,
        })
    return nodules


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Collect ground-truth nodule counts, diameters, and characteristics "
                     "from pylidc's LIDC-IDRI annotations for patients 1-20."
    )
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    patient_ids = [f"LIDC-IDRI-{i:04d}" for i in range(args.start, args.end + 1)]

    results = {}
    for n, patient_id in enumerate(patient_ids):
        scan = (pl.query(pl.Scan)
                  .filter(pl.Scan.patient_id == patient_id)
                  .first())
        if scan is None:
            print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: no scan found, skipping")
            results[patient_id] = {"found": False, "num_nodules": 0, "nodules": []}
            continue

        if not (DICOM_ROOT / patient_id).is_dir():
            print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: in pylidc DB but DICOM files "
                  f"not downloaded locally, skipping")
            results[patient_id] = {"found": False, "num_nodules": 0, "nodules": []}
            continue

        nodules = collect_scan(scan)
        results[patient_id] = {
            "found": True,
            "scan_id": scan.id,
            "series_instance_uid": scan.series_instance_uid,
            "num_nodules": len(nodules),
            "nodules": nodules,
        }
        print(f"[{n + 1}/{len(patient_ids)}] {patient_id}: {len(nodules)} nodule(s)")

    args.output.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.output}")
