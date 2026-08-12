"""Shared DICOM loading / cropping for the ground-truth-location characterization scripts
(5b_characterize_ground_truth_nodules.py and 5c_characterize_variants.py) - crop directly out
of the raw DICOM series at a nodule's own centroid_mm/diameter_mm, no detector involved. This
is the ground-truth-crop equivalent of roi_utils.py, which instead crops a detector box out of
the detector's resampled grid.
"""
import numpy as np
import pydicom


def load_patient_volume(patient_dir, series_uid):
    """Load one CT series into a HU volume (rows, cols, slices) plus the geometry
    needed to convert a world-mm point into an index into it: in-plane pixel
    spacing, the (x0, y0) ImagePositionPatient origin (constant across an axial
    series - same assumption 6_ground_truth_annotations.py's series_origin_xy
    makes), and each slice's own z position, sorted head-to-foot."""
    slices = []
    for f in patient_dir.rglob("*.dcm"):
        ds = pydicom.dcmread(f)
        if ds.Modality != "CT" or ds.SeriesInstanceUID != series_uid:
            continue
        slices.append(ds)
    slices.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))

    pixel_spacing = float(slices[0].PixelSpacing[0])  # assumed square, same as pylidc's own scan.pixel_spacing
    origin_xy = (float(slices[0].ImagePositionPatient[0]), float(slices[0].ImagePositionPatient[1]))
    z_positions = [float(ds.ImagePositionPatient[2]) for ds in slices]

    volume = np.stack([
        ds.pixel_array.astype(np.float32) * float(ds.get("RescaleSlope", 1)) + float(ds.get("RescaleIntercept", 0))
        for ds in slices
    ], axis=-1)
    return volume, pixel_spacing, origin_xy, z_positions


def crop_at_centroid(volume_hu, pixel_spacing, origin_xy, z_positions, centroid_mm, diameter_mm, pad_mm):
    """Crop a padded 3D region directly out of the original DICOM grid, centered
    on a world-mm point - the ground-truth equivalent of roi_utils.crop_roi,
    which instead crops around a detector box in the detector's resampled grid."""
    x0, y0 = origin_xy
    x, y, z = centroid_mm
    col = (x - x0) / pixel_spacing
    row = (y - y0) / pixel_spacing
    slice_idx = int(np.argmin(np.abs(np.array(z_positions) - z)))

    half_xy_px = max(1, round((diameter_mm / 2 + pad_mm) / pixel_spacing))
    slice_spacing = float(np.median(np.abs(np.diff(z_positions)))) if len(z_positions) > 1 else 1.0
    half_z = max(1, round((diameter_mm / 2 + pad_mm) / slice_spacing))

    rows, cols, depth = volume_hu.shape
    r0, r1 = max(0, round(row - half_xy_px)), min(rows, round(row + half_xy_px))
    c0, c1 = max(0, round(col - half_xy_px)), min(cols, round(col + half_xy_px))
    k0, k1 = max(0, slice_idx - half_z), min(depth, slice_idx + half_z + 1)
    return volume_hu[r0:r1, c0:c1, k0:k1]
