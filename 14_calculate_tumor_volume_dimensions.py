from pathlib import Path
import json
import csv
import numpy as np
import nibabel as nib


# ============================================================
# STEP 20 - TUMOR VOLUME AND DIMENSIONS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "ct_volumes"
)

MASK_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "tumor_ground_truth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "step14"
)

JSON_REPORT = (
    OUTPUT_DIR
    / "tumor_measurements.json"
)

CSV_REPORT = (
    OUTPUT_DIR
    / "tumor_measurements.csv"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 14- TUMOR VOLUME AND DIMENSIONS")
print("=" * 70)

print()
print("Measurement source:")
print("Original CT NIfTI + Original TCIA tumor ground truth")
print()
print(f"CT directory   : {CT_ROOT}")
print(f"Mask directory  : {MASK_ROOT}")


# ============================================================
# CHECK DIRECTORIES
# ============================================================

if not CT_ROOT.exists():
    raise SystemExit(
        f"ERROR: CT directory not found:\n{CT_ROOT}"
    )

if not MASK_ROOT.exists():
    raise SystemExit(
        f"ERROR: Tumor mask directory not found:\n{MASK_ROOT}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND PATIENTS
# ============================================================

patient_dirs = sorted(
    [
        p for p in MASK_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("Adrenal_Ki67_Seg_")
    ]
)

print()
print(f"Patients found : {len(patient_dirs)}")


if len(patient_dirs) != 53:
    print()
    print(
        "WARNING: Expected 53 patients."
    )


# ============================================================
# PROCESS PATIENTS
# ============================================================

results = []

successful = 0
failed = 0


for patient_dir in patient_dirs:

    patient_id = patient_dir.name

    mask_path = (
        patient_dir
        / f"{patient_id}_tumor_mask.nii.gz"
    )

    ct_path = (
        CT_ROOT
        / patient_id
        / f"{patient_id}_original_ct.nii.gz"
    )

    print()
    print("-" * 70)
    print(patient_id)

    try:

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Tumor mask not found:\n{mask_path}"
            )

        if not ct_path.exists():
            raise FileNotFoundError(
                f"CT volume not found:\n{ct_path}"
            )


        # ----------------------------------------------------
        # Load CT and mask
        # ----------------------------------------------------

        ct = nib.load(
            str(ct_path)
        )

        mask = nib.load(
            str(mask_path)
        )


        # ----------------------------------------------------
        # Geometry validation
        # ----------------------------------------------------

        if ct.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch: "
                f"CT={ct.shape}, Mask={mask.shape}"
            )

        ct_spacing = np.asarray(
            ct.header.get_zooms()[:3],
            dtype=np.float64
        )

        mask_spacing = np.asarray(
            mask.header.get_zooms()[:3],
            dtype=np.float64
        )

        spacing_difference = np.max(
            np.abs(
                ct_spacing
                - mask_spacing
            )
        )

        if spacing_difference > 1e-3:
            raise ValueError(
                "CT / mask spacing mismatch."
            )


        affine_difference = np.max(
            np.abs(
                ct.affine
                - mask.affine
            )
        )

        if affine_difference > 1e-3:
            raise ValueError(
                "CT / mask affine mismatch."
            )


        # ----------------------------------------------------
        # Read mask
        # ----------------------------------------------------

        mask_data = mask.get_fdata(
            dtype=np.float32
        )

        tumor = mask_data > 0.5

        tumor_voxels = int(
            np.count_nonzero(tumor)
        )

        if tumor_voxels == 0:
            raise ValueError(
                "Tumor mask is empty."
            )


        # ----------------------------------------------------
        # Voxel volume
        # ----------------------------------------------------

        voxel_volume_mm3 = float(
            np.prod(ct_spacing)
        )

        voxel_volume_cm3 = (
            voxel_volume_mm3 / 1000.0
        )

        tumor_volume_mm3 = (
            tumor_voxels
            * voxel_volume_mm3
        )

        tumor_volume_cm3 = (
            tumor_volume_mm3 / 1000.0
        )


        # ----------------------------------------------------
        # Tumor voxel coordinates
        # ----------------------------------------------------

        coordinates = np.argwhere(
            tumor
        )

        minimum = coordinates.min(
            axis=0
        )

        maximum = coordinates.max(
            axis=0
        )


        # ----------------------------------------------------
        # Physical bounding-box dimensions
        # ----------------------------------------------------

        bbox_voxel_size = (
            maximum
            - minimum
            + 1
        )

        bbox_mm = (
            bbox_voxel_size
            * ct_spacing
        )

        width_mm = float(
            bbox_mm[0]
        )

        height_mm = float(
            bbox_mm[1]
        )

        depth_mm = float(
            bbox_mm[2]
        )


        # ----------------------------------------------------
        # Maximum bounding-box dimension
        # ----------------------------------------------------

        maximum_diameter_mm = float(
            np.max(bbox_mm)
        )


        # ----------------------------------------------------
        # Tumor centroid in voxel coordinates
        # ----------------------------------------------------

        centroid_voxel = (
            coordinates.mean(
                axis=0
            )
        )


        # ----------------------------------------------------
        # Tumor centroid in world coordinates
        # ----------------------------------------------------

        centroid_world = nib.affines.apply_affine(
            ct.affine,
            centroid_voxel
        )


        # ----------------------------------------------------
        # Tumor slice range
        # ----------------------------------------------------

        z_min = int(
            minimum[2]
        )

        z_max = int(
            maximum[2]
        )

        slice_count = (
            z_max
            - z_min
            + 1
        )


        # ----------------------------------------------------
        # Maximum area slice
        # ----------------------------------------------------

        slice_areas = np.sum(
            tumor,
            axis=(0, 1)
        )

        maximum_area_slice = int(
            np.argmax(slice_areas)
        )

        maximum_area_voxels = int(
            slice_areas[
                maximum_area_slice
            ]
        )

        maximum_area_mm2 = (
            maximum_area_voxels
            * ct_spacing[0]
            * ct_spacing[1]
        )


        # ----------------------------------------------------
        # Result
        # ----------------------------------------------------

        result = {

            "patient_id": patient_id,

            "tumor_voxels": tumor_voxels,

            "voxel_spacing_mm": [
                float(x)
                for x in ct_spacing
            ],

            "voxel_volume_mm3":
                voxel_volume_mm3,

            "tumor_volume_mm3":
                tumor_volume_mm3,

            "tumor_volume_cm3":
                tumor_volume_cm3,

            "bounding_box": {

                "voxel_min": [
                    int(x)
                    for x in minimum
                ],

                "voxel_max": [
                    int(x)
                    for x in maximum
                ],

                "voxel_size": [
                    int(x)
                    for x in bbox_voxel_size
                ],

                "width_mm":
                    width_mm,

                "height_mm":
                    height_mm,

                "depth_mm":
                    depth_mm,

                "maximum_diameter_mm":
                    maximum_diameter_mm
            },

            "centroid": {

                "voxel": [
                    float(x)
                    for x in centroid_voxel
                ],

                "world_mm": [
                    float(x)
                    for x in centroid_world
                ]
            },

            "tumor_slice_range": {

                "first_slice":
                    z_min,

                "last_slice":
                    z_max,

                "slice_count":
                    slice_count
            },

            "maximum_area_slice": {

                "slice_index":
                    maximum_area_slice,

                "tumor_pixels":
                    maximum_area_voxels,

                "area_mm2":
                    float(maximum_area_mm2)
            },

            "geometry_validation": {

                "shape_match":
                    True,

                "spacing_match":
                    True,

                "spacing_difference_mm":
                    float(spacing_difference),

                "affine_match":
                    True,

                "affine_difference":
                    float(affine_difference)
            },

            "status":
                "PASS"
        }


        results.append(
            result
        )

        successful += 1


        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

        print(
            f"Tumor voxels        : "
            f"{tumor_voxels:,}"
        )

        print(
            f"Voxel spacing       : "
            f"{ct_spacing[0]:.6f} x "
            f"{ct_spacing[1]:.6f} x "
            f"{ct_spacing[2]:.6f} mm"
        )

        print(
            f"Tumor volume        : "
            f"{tumor_volume_cm3:.3f} cm3"
        )

        print(
            f"Bounding box        : "
            f"{width_mm:.2f} x "
            f"{height_mm:.2f} x "
            f"{depth_mm:.2f} mm"
        )

        print(
            f"Maximum diameter    : "
            f"{maximum_diameter_mm:.2f} mm"
        )

        print(
            f"Tumor slices        : "
            f"{z_min} - {z_max}"
        )

        print(
            f"Max area slice      : "
            f"{maximum_area_slice}"
        )

        print(
            f"Geometry validation : PASS"
        )


    except Exception as exc:

        failed += 1

        print(
            f"ERROR: {exc}"
        )

        results.append(
            {
                "patient_id":
                    patient_id,

                "status":
                    "FAIL",

                "error":
                    str(exc)
            }
        )


# ============================================================
# SAVE JSON
# ============================================================

report = {

    "step": 14,

    "purpose":
        "Calculate tumor volume and physical dimensions "
        "from original CT geometry and original TCIA "
        "tumor ground-truth masks.",

    "measurement_source": {

        "ct":
            str(CT_ROOT),

        "tumor_ground_truth":
            str(MASK_ROOT)
    },

    "patients_found":
        len(patient_dirs),

    "successful":
        successful,

    "failed":
        failed,

    "results":
        results,

    "status":
        "PASS" if failed == 0 else "FAIL"
}


with open(
    JSON_REPORT,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# SAVE CSV
# ============================================================

csv_rows = []

for result in results:

    if result.get("status") != "PASS":
        csv_rows.append(
            {
                "patient_id":
                    result["patient_id"],
                "status":
                    "FAIL",
                "error":
                    result.get("error", "")
            }
        )

        continue

    bbox = result["bounding_box"]

    csv_rows.append(
        {
            "patient_id":
                result["patient_id"],

            "tumor_voxels":
                result["tumor_voxels"],

            "tumor_volume_cm3":
                result["tumor_volume_cm3"],

            "width_mm":
                bbox["width_mm"],

            "height_mm":
                bbox["height_mm"],

            "depth_mm":
                bbox["depth_mm"],

            "maximum_diameter_mm":
                bbox["maximum_diameter_mm"],

            "first_slice":
                result["tumor_slice_range"][
                    "first_slice"
                ],

            "last_slice":
                result["tumor_slice_range"][
                    "last_slice"
                ],

            "slice_count":
                result["tumor_slice_range"][
                    "slice_count"
                ],

            "maximum_area_slice":
                result["maximum_area_slice"][
                    "slice_index"
                ],

            "maximum_area_mm2":
                result["maximum_area_slice"][
                    "area_mm2"
                ],

            "status":
                "PASS"
        }
    )


if csv_rows:

    fieldnames = list(
        csv_rows[0].keys()
    )

    with open(
        CSV_REPORT,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            csv_rows
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 14 SUMMARY")
print("=" * 70)

print(
    f"Patients found     : {len(patient_dirs)}"
)

print(
    f"Successful         : {successful}"
)

print(
    f"Failed             : {failed}"
)

print()
print(
    f"JSON report        : {JSON_REPORT}"
)

print(
    f"CSV report         : {CSV_REPORT}"
)

if failed == 0:

    print()
    print(
        "STEP 14 STATUS: "
        "TUMOR VOLUME AND DIMENSIONS CALCULATED SUCCESSFULLY"
    )

else:

    print()
    print(
        "STEP 14 STATUS: "
        "COMPLETED WITH ERRORS"
    )

print("=" * 70)