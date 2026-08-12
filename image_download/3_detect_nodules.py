import argparse
import time

from nodule_detector import DICOM_ROOT, build_detector, detect_patient

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", help="Patient ID, e.g. LIDC-IDRI-0001. Defaults to the first patient found.")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--device", default="cpu", help="'cpu' or 'mps'. Box selection/NMS still run on CPU either way.")
    args = ap.parse_args()

    print(f"Loading detector (network on {args.device})...")
    detector, preprocessing, postprocessing = build_detector(args.device)

    if args.patient:
        patient_dir = DICOM_ROOT / args.patient
    else:
        patient_dir = sorted(d for d in DICOM_ROOT.iterdir() if d.is_dir())[0]

    print(f"Running detection on {patient_dir.name}...")
    start = time.monotonic()
    num_slices, world_boxes, scores, voxel_boxes, volume = detect_patient(
        patient_dir, detector, preprocessing, postprocessing, args.device, args.score_thresh
    )
    elapsed = time.monotonic() - start

    print(f"{patient_dir.name}: {num_slices} slices -> {len(world_boxes)} nodule(s) at score>={args.score_thresh}"
          f"  ({elapsed:.1f}s)")
    for box, score in zip(world_boxes.tolist(), scores.tolist()):
        cx, cy, cz, w, h, d = box
        print(f"  center=({cx:.1f}, {cy:.1f}, {cz:.1f})mm  size=({w:.1f}x{h:.1f}x{d:.1f})mm  score={score:.3f}")
