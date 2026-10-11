from pathlib import Path
import os
import json
import numpy as np
import nibabel as nib

# ============================================================
# CONFIGURATION
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parent


CT_ROOT = os.path.join(
    PROJECT_ROOT , "dataset", "ct_volumes"
)

MASK_ROOT = os.path.join(
    PROJECT_ROOT , "dataset", "tumor_ground_truth"
)

REPORT_DIR = os.path.join(
    PROJECT_ROOT , "dataset", "metadata", "step06"
)

REPORT_FILE = os.path.join(
    REPORT_DIR, "ct_tumor_alignment_report.json"
)

os.makedirs(REPORT_DIR, exist_ok=True)


# ============================================================
# PATIENTS
# ============================================================

patients = sorted(
    p for p in os.listdir(CT_ROOT)
    if p.startswith("Adrenal_Ki67_Seg_")
    and os.path.isdir(os.path.join(CT_ROOT, p))
)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("STEP 06 - CT / TUMOR GROUND-TRUTH SPATIAL VALIDATION")
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
        # FILES
        # ----------------------------------------------------

        ct_file = os.path.join(
            CT_ROOT,
            patient,
            f"{patient}_original_ct.nii.gz"
        )

        mask_file = os.path.join(
            MASK_ROOT,
            patient,
            f"{patient}_tumor_mask.nii.gz"
        )

        if not os.path.isfile(ct_file):
            raise RuntimeError("CT volume not found")

        if not os.path.isfile(mask_file):
            raise RuntimeError("Tumor mask not found")

        # ----------------------------------------------------
        # LOAD NIFTI
        # ----------------------------------------------------

        ct = nib.load(ct_file)
        mask = nib.load(mask_file)

        ct_shape = tuple(ct.shape[:3])
        mask_shape = tuple(mask.shape[:3])

        ct_spacing = np.asarray(
            ct.header.get_zooms()[:3],
            dtype=float
        )

        mask_spacing = np.asarray(
            mask.header.get_zooms()[:3],
            dtype=float
        )

        # ----------------------------------------------------
        # GEOMETRY
        # ----------------------------------------------------

        shape_match = ct_shape == mask_shape

        spacing_match = np.allclose(
            ct_spacing,
            mask_spacing,
            atol=1e-3,
            rtol=0
        )

        affine_difference = float(
            np.max(
                np.abs(
                    ct.affine - mask.affine
                )
            )
        )

        affine_match = affine_difference <= 1e-3

        # ----------------------------------------------------
        # MASK VALIDATION
        # ----------------------------------------------------

        mask_data = mask.get_fdata(dtype=np.float32)

        unique = np.unique(mask_data)

        binary = bool(
            np.all(
                np.isin(unique, [0, 1])
            )
        )

        tumor_voxels = int(
            np.count_nonzero(mask_data > 0)
        )

        non_empty = tumor_voxels > 0

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        voxel_volume_mm3 = float(
            np.prod(mask_spacing)
        )

        tumor_volume_cm3 = (
            tumor_voxels *
            voxel_volume_mm3 /
            1000.0
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        valid = (
            shape_match
            and spacing_match
            and affine_match
            and binary
            and non_empty
        )

        status = "PASS" if valid else "FAIL"

        if valid:
            passed += 1
        else:
            failed += 1

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        print(
            "  CT shape       :", ct_shape
        )

        print(
            "  Mask shape     :", mask_shape
        )

        print(
            "  Spacing match  :", spacing_match
        )

        print(
            "  Affine match   :", affine_match
        )

        print(
            "  Affine diff    :", affine_difference
        )

        print(
            "  Binary mask    :", binary
        )

        print(
            "  Tumor voxels   :", tumor_voxels
        )

        print(
            "  Tumor volume   :",
            f"{tumor_volume_cm3:.3f} cm3"
        )

        print(
            "  STATUS         :", status
        )

        print()

        reports.append({
            "patient": patient,
            "ct_file": ct_file,
            "tumor_mask": mask_file,
            "ct_shape": list(ct_shape),
            "mask_shape": list(mask_shape),
            "shape_match": shape_match,
            "ct_spacing_mm": ct_spacing.tolist(),
            "mask_spacing_mm": mask_spacing.tolist(),
            "spacing_match": spacing_match,
            "affine_max_difference": affine_difference,
            "affine_match": affine_match,
            "mask_unique_values": unique.tolist(),
            "binary_mask": binary,
            "tumor_voxels": tumor_voxels,
            "tumor_volume_cm3": tumor_volume_cm3,
            "status": status
        })

        del mask_data

    except Exception as e:

        failed += 1

        print("  ERROR:", e)
        print("  STATUS: FAIL")
        print()

        reports.append({
            "patient": patient,
            "status": "FAIL",
            "error": str(e)
        })


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        reports,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("STEP 06 COMPLETE")
print("=" * 70)

print(
    "Patients inspected     :",
    len(patients)
)

print(
    "Alignment PASS         :",
    passed
)

print(
    "Alignment FAIL         :",
    failed
)

print(
    "Report                 :",
    REPORT_FILE
)

print()

if failed == 0:

    print(
        "STEP 06 STATUS: ALL CT / TUMOR MASKS SPATIALLY VALIDATED"
    )

else:

    print(
        "STEP 06 STATUS: REVIEW REQUIRED"
    )