import tempfile
from pathlib import Path

import pydicom
import SimpleITK as sitk
import torch
from monai.bundle import ConfigParser

BUNDLE_ROOT = Path(__file__).resolve().parent / "monai_bundles" / "lung_nodule_ct_detection"
DICOM_ROOT = Path(__file__).resolve().parent.parent / "datasets/LDIC-IDRI-subset/lidc_idri"


def largest_ct_series(patient_dir):
    """Group a patient's CT DICOM files (which may be nested in subdirectories) by
    SeriesInstanceUID and return the largest one's file paths ordered by z-position.

    SimpleITK's GDCM series discovery only scans one flat directory, which doesn't match
    how LIDC-IDRI nests files under study/series subfolders - so series grouping and
    ordering is done directly with pydicom instead, the same way the rest of this
    pipeline already does it (see image_conversion.py / model_eval.py).
    """
    series = {}
    for f in patient_dir.rglob("*.dcm"):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        if ds.Modality != "CT":
            continue
        position = ds.get("ImagePositionPatient")
        z = float(position[2]) if position else float(ds.get("SliceLocation", 0.0))
        series.setdefault(ds.SeriesInstanceUID, []).append((z, f))

    largest_uid = max(series, key=lambda uid: len(series[uid]))
    return [str(f) for _, f in sorted(series[largest_uid])]


def series_to_nifti(patient_dir, out_path):
    """Read one patient's CT DICOM series and write it as a NIfTI volume."""
    file_names = largest_ct_series(patient_dir)
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(file_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(out_path))
    return len(file_names)


def build_detector(device):
    parser = ConfigParser()
    parser.read_config(str(BUNDLE_ROOT / "configs" / "inference.json"))
    parser["bundle_root"] = str(BUNDLE_ROOT)
    parser["whether_raw_luna16"] = True  # our volume isn't pre-resampled; let the bundle resample it
    parser["device"] = device
    # The bundle's default [512, 512, 192] patch is ~the whole volume in one sliding-window
    # pass through a 3D ResNet50-FPN - fine on the datacenter GPU it was designed for, but it
    # OOM-kills a 16GB CPU machine. Trade more (smaller, overlapping) windows for lower peak memory.
    parser["infer_patch_size"] = [192, 192, 80]
    parser.parse()

    preprocessing = parser.get_parsed_content("preprocessing")
    postprocessing = parser.get_parsed_content("postprocessing")
    detector = parser.get_parsed_content("detector")
    parser.get_parsed_content("detector_ops")  # executes the $-statements that configure detector thresholds/inferer
    network = parser.get_parsed_content("network")

    ckpt = torch.load(BUNDLE_ROOT / "models" / "model.pt", map_location=device)
    network.load_state_dict(ckpt)
    detector.eval()
    return detector, preprocessing, postprocessing


def detect_patient(patient_dir, detector, preprocessing, postprocessing, device, score_thresh):
    """Returns:
    - num_slices: slice count of the source series
    - world_boxes/scores: detections in world mm (center x,y,z, width,height,depth) - for reporting
    - voxel_boxes: the same kept detections in xyzxyz voxel-index space of `volume` - for cropping
    - volume: the preprocessed (oriented + resampled) volume tensor, shape (1, H, W, D), intensity
      scaled to [0, 1] over the bundle's [-1024, 300] HU window (see ScaleIntensityRanged in
      configs/inference.json) - invert with `hu = volume * 1324.0 - 1024.0` to recover HU values.
    """
    with tempfile.TemporaryDirectory() as tmp:
        nifti_path = Path(tmp) / "volume.nii.gz"
        num_slices = series_to_nifti(patient_dir, nifti_path)

        data = preprocessing({"image": str(nifti_path)})
        image = data["image"].to(device).unsqueeze(0)  # add batch dim -> (1, C, H, W, D)

        with torch.no_grad():
            outputs = detector(image, use_inferer=True)
        pred = outputs[0]

        # Filter by score first (in the raw, pre-postprocessing voxel-space box order) so the
        # voxel-space boxes we return for cropping stay index-aligned with the world-mm boxes -
        # ClipBoxToImaged below only clips/filters "box"+"label", not "label_scores", so filtering
        # after postprocessing could desync the two.
        scores = pred["label_scores"]
        order = scores.argsort(descending=True)
        keep = order[scores[order] >= score_thresh]
        voxel_boxes = pred["box"][keep]
        scores = scores[keep]

        result = postprocessing({
            "box": voxel_boxes,
            "label": pred["label"][keep],
            "label_scores": scores,
            "image": data["image"],
        })

    return num_slices, result["box"], scores, voxel_boxes, data["image"]
