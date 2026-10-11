from pathlib import Path
import os
import json
import numpy as np
import nibabel as nib
import pydicom

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TCIA = os.path.join(PROJECT_ROOT, "raw", "adrenal_acc_ki67_seg")
OUT = os.path.join(PROJECT_ROOT, "dataset", "tumor_ground_truth")
REPORT = os.path.join(
    PROJECT_ROOT, "dataset", "metadata", "step05",
    "tumor_ground_truth_report.json"
)

os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.dirname(REPORT), exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def read(path, pixels=False):
    try:
        return pydicom.dcmread(
            path, force=True,
            stop_before_pixels=not pixels
        )
    except Exception:
        return None


def find_seg(patient):
    root = os.path.join(TCIA, patient)
    for dp, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dp, f)
            ds = read(p)
            if ds is not None and getattr(ds, "Modality", "") == "SEG":
                return p
    return None


def tumor_segment(seg):
    for s in getattr(seg, "SegmentSequence", []):
        text = (
            str(getattr(s, "SegmentLabel", "")) + " " +
            str(getattr(s, "SegmentDescription", ""))
        ).lower()

        if any(x in text for x in
               ("tumor", "tumour", "mass", "lesion", "neoplasm")):
            return int(s.SegmentNumber)
    return None


def frame_segment(frame):
    try:
        return int(
            frame.SegmentIdentificationSequence[0]
            .ReferencedSegmentNumber
        )
    except Exception:
        return None


def frame_uid(frame):
    try:
        return str(
            frame.DerivationImageSequence[0]
            .SourceImageSequence[0]
            .ReferencedSOPInstanceUID
        )
    except Exception:
        return None


def referenced_uids(seg):
    result = set()

    for series in getattr(seg, "ReferencedSeriesSequence", []):
        for item in getattr(
            series, "ReferencedInstanceSequence", []
        ):
            uid = getattr(
                item, "ReferencedSOPInstanceUID", None
            )
            if uid:
                result.add(str(uid))

    return result


def ct_lookup(patient, wanted):
    result = {}
    root = os.path.join(TCIA, patient)

    for dp, _, files in os.walk(root):
        for f in files:
            p = os.path.join(dp, f)
            ds = read(p)

            if ds is None or getattr(ds, "Modality", "") != "CT":
                continue

            uid = getattr(ds, "SOPInstanceUID", None)

            if uid and str(uid) in wanted:
                result[str(uid)] = ds

    return result


# ============================================================
# MAIN
# ============================================================

patients = sorted(
    p for p in os.listdir(TCIA)
    if p.startswith("Adrenal_Ki67_Seg_")
    and os.path.isdir(os.path.join(TCIA, p))
)

print("=" * 70)
print("STEP 05 - ORIGINAL TCIA TUMOR GROUND TRUTH")
print("=" * 70)
print("Patients:", len(patients))
print()

reports = []
passed = 0
failed = 0

for n, patient in enumerate(patients, 1):

    print(f"[{n:02d}/{len(patients):02d}] {patient}")

    try:
        # ----------------------------------------------------
        # SEG
        # ----------------------------------------------------

        seg_path = find_seg(patient)

        if not seg_path:
            raise RuntimeError("SEG not found")

        seg = read(seg_path, pixels=True)

        if seg is None:
            raise RuntimeError("Unable to read SEG")

        segment = tumor_segment(seg)

        if segment is None:
            raise RuntimeError("Tumor segment not found")

        frames = seg.PerFrameFunctionalGroupsSequence
        pixels = seg.pixel_array

        # ----------------------------------------------------
        # EXACT CT REFERENCES
        # ----------------------------------------------------

        refs = referenced_uids(seg)
        ct = ct_lookup(patient, refs)

        if len(ct) != len(refs):
            raise RuntimeError(
                f"Referenced CT missing: {len(ct)}/{len(refs)}"
            )

        # ----------------------------------------------------
        # CT GEOMETRY
        # ----------------------------------------------------

        order = sorted(
            ct,
            key=lambda u: float(
                ct[u].ImagePositionPatient[2]
            )
        )

        first = ct[order[0]]

        rows = int(first.Rows)
        cols = int(first.Columns)

        spacing = np.asarray(
            first.PixelSpacing, dtype=float
        )

        if len(order) > 1:
            p0 = np.asarray(
                ct[order[0]].ImagePositionPatient,
                dtype=float
            )
            p1 = np.asarray(
                ct[order[1]].ImagePositionPatient,
                dtype=float
            )
            slice_spacing = float(np.linalg.norm(p1 - p0))
        else:
            slice_spacing = float(
                getattr(first, "SliceThickness", 1.0)
            )

        uid_index = {
            uid: i for i, uid in enumerate(order)
        }

        # ----------------------------------------------------
        # MASK
        # ----------------------------------------------------

        mask = np.zeros(
            (rows, cols, len(order)),
            dtype=np.uint8
        )

        mapped = 0
        skipped_unreferenced = 0

        for i, frame in enumerate(frames):

            if frame_segment(frame) != segment:
                continue

            uid = frame_uid(frame)

            # Only exact CT SOP references are accepted.
            if uid not in uid_index:
                skipped_unreferenced += 1
                continue

            z = uid_index[uid]
            fp = pixels[i]

            if fp.shape != (rows, cols):
                raise RuntimeError(
                    f"Invalid SEG frame shape: {fp.shape}"
                )

            mask[:, :, z] = np.maximum(
                mask[:, :, z],
                (fp > 0).astype(np.uint8)
            )

            mapped += 1

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        voxels = int(np.count_nonzero(mask))

        if voxels == 0:
            raise RuntimeError("Tumor mask is empty")

        orientation = np.asarray(
            first.ImageOrientationPatient,
            dtype=float
        )

        row = orientation[:3]
        col = orientation[3:]
        normal = np.cross(row, col)

        origin = np.asarray(
            first.ImagePositionPatient,
            dtype=float
        )

        affine = np.eye(4)

        affine[:3, 0] = row * spacing[0]
        affine[:3, 1] = col * spacing[1]
        affine[:3, 2] = normal * slice_spacing
        affine[:3, 3] = origin

        img = nib.Nifti1Image(mask, affine)
        img.header.set_data_dtype(np.uint8)

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        out_dir = os.path.join(OUT, patient)
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(
            out_dir,
            f"{patient}_tumor_mask.nii.gz"
        )

        nib.save(img, out_file)

        # ----------------------------------------------------
        # REPORT
        # ----------------------------------------------------

        record = {
            "patient": patient,
            "seg_file": seg_path,
            "segment_number": segment,
            "seg_frames": len(frames),
            "referenced_ct_slices": len(refs),
            "ct_slices_used": len(order),
            "tumor_frames_mapped": mapped,
            "unreferenced_tumor_frames": skipped_unreferenced,
            "tumor_voxels": voxels,
            "shape": list(mask.shape),
            "spacing_mm": [
                float(spacing[0]),
                float(spacing[1]),
                float(slice_spacing)
            ],
            "output": out_file,
            "status": "PASS"
        }

        reports.append(record)
        passed += 1

        print("  CT slices       :", len(order))
        print("  SEG frames      :", len(frames))
        print("  Frames mapped   :", mapped)
        print("  Unreferenced    :", skipped_unreferenced)
        print("  Tumor voxels    :", voxels)
        print("  STATUS          : PASS")
        print()

    except Exception as e:

        failed += 1

        reports.append({
            "patient": patient,
            "status": "FAIL",
            "error": str(e)
        })

        print("  ERROR:", e)
        print("  STATUS: FAIL")
        print()


# ============================================================
# REPORT
# ============================================================

with open(REPORT, "w", encoding="utf-8") as f:
    json.dump(reports, f, indent=2)

print("=" * 70)
print("STEP 05 COMPLETE")
print("=" * 70)
print("Patients inspected     :", len(patients))
print("Successful extractions :", passed)
print("Failed                 :", failed)
print("Report                 :", REPORT)

if failed == 0:
    print("STEP 05 STATUS: ALL PATIENT MASKS EXTRACTED")
else:
    print("STEP 05 STATUS: REVIEW REQUIRED")
