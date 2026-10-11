from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pydicom
import nibabel as nib


# ============================================================
# STEP 04
# Reconstruct CT volumes using ONLY the CT SOP Instance UIDs
# explicitly referenced by the original DICOM SEG.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TCIA_ROOT = (
    PROJECT_ROOT
    / "raw"
    / "adrenal_acc_ki67_seg"
)

STEP03_REPORT = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "step03"
    / "seg_referenced_ct_series.json"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "ct_volumes"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "step04"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_FILE = (
    REPORT_DIR
    / "ct_reconstruction_report.json"
)


# ------------------------------------------------------------
# DICOM reader
# ------------------------------------------------------------

def read_dicom(path):
    try:
        return pydicom.dcmread(
            str(path),
            force=False
        )
    except Exception:
        return None


# ------------------------------------------------------------
# Extract referenced CT SOP Instance UIDs directly from SEG
# ------------------------------------------------------------

def get_referenced_ct_uids(seg_ds):

    referenced_uids = set()

    if not hasattr(
        seg_ds,
        "ReferencedSeriesSequence"
    ):
        return referenced_uids

    for series_item in seg_ds.ReferencedSeriesSequence:

        if not hasattr(
            series_item,
            "ReferencedInstanceSequence"
        ):
            continue

        for instance in (
            series_item.ReferencedInstanceSequence
        ):

            if hasattr(
                instance,
                "ReferencedSOPInstanceUID"
            ):

                referenced_uids.add(
                    str(
                        instance.ReferencedSOPInstanceUID
                    )
                )

    return referenced_uids


# ------------------------------------------------------------
# Get orientation vectors
# ------------------------------------------------------------

def get_orientation_vectors(ds):

    iop = np.asarray(
        [
            float(x)
            for x in ds.ImageOrientationPatient
        ],
        dtype=np.float64
    )

    row_direction = iop[:3]
    column_direction = iop[3:]

    slice_direction = np.cross(
        row_direction,
        column_direction
    )

    return (
        row_direction,
        column_direction,
        slice_direction
    )


# ------------------------------------------------------------
# Calculate spatial slice coordinate
# ------------------------------------------------------------

def get_slice_coordinate(
    ds,
    slice_direction
):

    position = np.asarray(
        [
            float(x)
            for x in ds.ImagePositionPatient
        ],
        dtype=np.float64
    )

    return float(
        np.dot(
            position,
            slice_direction
        )
    )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "STEP 04 - ORIGINAL CT VOLUME RECONSTRUCTION"
    )
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nTCIA dataset:")
    print(TCIA_ROOT)

    print("\nStep 03 report:")
    print(STEP03_REPORT)

    if not TCIA_ROOT.exists():

        print(
            "\nERROR: TCIA dataset folder not found."
        )

        return

    if not STEP03_REPORT.exists():

        print(
            "\nERROR: Step 03 report not found."
        )

        return

    # --------------------------------------------------------
    # Load Step 03 report
    # --------------------------------------------------------

    with open(
        STEP03_REPORT,
        "r",
        encoding="utf-8"
    ) as f:

        step03_data = json.load(f)

    patient_reports = step03_data.get(
        "reports",
        []
    )

    print(
        f"\nPatients in Step 03 report: "
        f"{len(patient_reports)}"
    )

    reconstruction_reports = []

    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process patients
    # --------------------------------------------------------

    for index, patient_report in enumerate(
        patient_reports,
        start=1
    ):

        patient_id = patient_report[
            "patient_id"
        ]

        print("\n" + "-" * 70)

        print(
            f"[{index}/{len(patient_reports)}] "
            f"{patient_id}"
        )

        print("-" * 70)

        patient_dir = (
            TCIA_ROOT
            / patient_id
        )

        if not patient_dir.exists():

            print(
                "ERROR: Patient folder not found."
            )

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "status":
                    "PATIENT_FOLDER_NOT_FOUND"

            })

            continue

        # ----------------------------------------------------
        # Get SEG file
        # ----------------------------------------------------

        seg_file = Path(
            patient_report["seg_file"]
        )

        if not seg_file.exists():

            print(
                "ERROR: SEG file not found:"
            )

            print(seg_file)

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "status":
                    "SEG_FILE_NOT_FOUND"

            })

            continue

        print("\nSEG file:")
        print(seg_file)

        # ----------------------------------------------------
        # Read SEG
        # ----------------------------------------------------

        seg_ds = read_dicom(
            seg_file
        )

        if seg_ds is None:

            print(
                "ERROR: Could not read SEG."
            )

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "status":
                    "SEG_READ_ERROR"

            })

            continue

        # ----------------------------------------------------
        # Extract exact CT references
        # ----------------------------------------------------

        referenced_uids = (
            get_referenced_ct_uids(
                seg_ds
            )
        )

        expected_count = len(
            referenced_uids
        )

        print(
            "\nReferenced CT SOP Instance UIDs:"
        )

        print(
            expected_count
        )

        if expected_count == 0:

            print(
                "ERROR: SEG contains no referenced CT instances."
            )

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "status":
                    "NO_REFERENCED_CT_UIDS"

            })

            continue

        # ----------------------------------------------------
        # Build CT SOP UID lookup
        #
        # IMPORTANT:
        # We scan all CT objects, but later select ONLY
        # the SOP Instance UIDs referenced by the SEG.
        # ----------------------------------------------------

        ct_lookup = {}

        scanned = 0
        readable = 0
        ct_count = 0

        for path in patient_dir.rglob("*"):

            if not path.is_file():
                continue

            scanned += 1

            ds = read_dicom(path)

            if ds is None:
                continue

            readable += 1

            modality = str(
                getattr(
                    ds,
                    "Modality",
                    ""
                )
            ).upper()

            if modality != "CT":
                continue

            ct_count += 1

            sop_uid = str(
                getattr(
                    ds,
                    "SOPInstanceUID",
                    ""
                )
            )

            if not sop_uid:
                continue

            ct_lookup[sop_uid] = (
                path,
                ds
            )

        print(
            f"\nReadable DICOM objects: "
            f"{readable}"
        )

        print(
            f"Total CT objects: "
            f"{ct_count}"
        )

        # ----------------------------------------------------
        # Locate ONLY referenced CT files
        # ----------------------------------------------------

        referenced_ct = []

        missing_uids = []

        for sop_uid in sorted(
            referenced_uids
        ):

            if sop_uid in ct_lookup:

                path, ds = ct_lookup[
                    sop_uid
                ]

                referenced_ct.append(
                    (
                        path,
                        ds
                    )
                )

            else:

                missing_uids.append(
                    sop_uid
                )

        print(
            f"Referenced CT files located: "
            f"{len(referenced_ct)}"
        )

        print(
            f"Referenced CT files missing: "
            f"{len(missing_uids)}"
        )

        if missing_uids:

            print(
                "\nERROR: Some referenced CT files "
                "could not be located."
            )

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "referenced_ct_count":
                    expected_count,

                "located_ct_count":
                    len(referenced_ct),

                "missing_ct_count":
                    len(missing_uids),

                "missing_sop_instance_uids":
                    missing_uids,

                "status":
                    "REFERENCED_CT_FILES_MISSING"

            })

            continue

        # ----------------------------------------------------
        # Verify SeriesInstanceUID
        # ----------------------------------------------------

        series_uids = sorted(
            set(
                str(
                    getattr(
                        ds,
                        "SeriesInstanceUID",
                        ""
                    )
                )
                for _, ds in referenced_ct
            )
        )

        print(
            f"\nReferenced CT Series count: "
            f"{len(series_uids)}"
        )

        for series_uid in series_uids:

            print(
                f"  {series_uid}"
            )

        if len(series_uids) != 1:

            print(
                "\nERROR: Referenced CT instances "
                "belong to multiple CT series."
            )

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "series_count":
                    len(series_uids),

                "series_uids":
                    series_uids,

                "status":
                    "MULTIPLE_REFERENCED_CT_SERIES"

            })

            continue

        series_uid = series_uids[0]

        # ----------------------------------------------------
        # Get reference geometry
        # ----------------------------------------------------

        first_ds = referenced_ct[0][1]

        (
            row_direction,
            column_direction,
            slice_direction
        ) = get_orientation_vectors(
            first_ds
        )

        # ----------------------------------------------------
        # Verify all orientations
        # ----------------------------------------------------

        orientation_ok = True

        reference_iop = np.asarray(
            [
                float(x)
                for x in first_ds.ImageOrientationPatient
            ],
            dtype=np.float64
        )

        for _, ds in referenced_ct:

            current_iop = np.asarray(
                [
                    float(x)
                    for x in ds.ImageOrientationPatient
                ],
                dtype=np.float64
            )

            if not np.allclose(
                current_iop,
                reference_iop,
                atol=1e-5
            ):

                orientation_ok = False
                break

        if not orientation_ok:

            print(
                "ERROR: Referenced CT orientations "
                "are inconsistent."
            )

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "status":
                    "ORIENTATION_INCONSISTENT"

            })

            continue

        # ----------------------------------------------------
        # Sort referenced slices spatially
        # ----------------------------------------------------

        slice_entries = []

        for path, ds in referenced_ct:

            coordinate = (
                get_slice_coordinate(
                    ds,
                    slice_direction
                )
            )

            slice_entries.append({

                "path": path,

                "ds": ds,

                "coordinate":
                    coordinate

            })

        slice_entries.sort(
            key=lambda x:
            x["coordinate"]
        )

        # ----------------------------------------------------
        # Check duplicate spatial positions
        # ----------------------------------------------------

        coordinates = np.asarray(
            [
                entry["coordinate"]
                for entry in slice_entries
            ],
            dtype=np.float64
        )

        duplicate_positions = 0

        if len(coordinates) > 1:

            differences = np.diff(
                coordinates
            )

            duplicate_positions = int(
                np.sum(
                    np.isclose(
                        differences,
                        0.0,
                        atol=1e-5
                    )
                )
            )

        if duplicate_positions > 0:

            print(
                "ERROR: Duplicate slice positions detected."
            )

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "status":
                    "DUPLICATE_SLICE_POSITIONS"

            })

            continue

        # ----------------------------------------------------
        # Calculate slice spacing
        # ----------------------------------------------------

        if len(coordinates) > 1:

            differences = np.diff(
                coordinates
            )

            spacing_values = np.abs(
                differences
            )

            median_spacing = float(
                np.median(
                    spacing_values
                )
            )

            min_spacing = float(
                np.min(
                    spacing_values
                )
            )

            max_spacing = float(
                np.max(
                    spacing_values
                )
            )

        else:

            median_spacing = 0.0
            min_spacing = 0.0
            max_spacing = 0.0

        # ----------------------------------------------------
        # Image dimensions
        # ----------------------------------------------------

        rows = int(
            first_ds.Rows
        )

        columns = int(
            first_ds.Columns
        )

        pixel_spacing = [
            float(x)
            for x in first_ds.PixelSpacing
        ]

        row_spacing = (
            pixel_spacing[0]
        )

        column_spacing = (
            pixel_spacing[1]
        )

        # ----------------------------------------------------
        # Verify dimensions and pixel spacing
        # ----------------------------------------------------

        geometry_ok = True

        for _, ds in referenced_ct:

            if int(ds.Rows) != rows:
                geometry_ok = False

            if int(ds.Columns) != columns:
                geometry_ok = False

            current_spacing = [
                float(x)
                for x in ds.PixelSpacing
            ]

            if not np.allclose(
                current_spacing,
                pixel_spacing,
                atol=1e-5
            ):

                geometry_ok = False

        if not geometry_ok:

            print(
                "ERROR: Referenced CT geometry "
                "is inconsistent."
            )

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "status":
                    "REFERENCED_CT_GEOMETRY_INCONSISTENT"

            })

            continue

        # ----------------------------------------------------
        # Read CT pixel data
        # ----------------------------------------------------

        volume_slices = []

        try:

            for entry in slice_entries:

                ds = entry["ds"]

                stored_pixels = (
                    ds.pixel_array
                )

                stored_pixels = (
                    stored_pixels.astype(
                        np.float32
                    )
                )

                slope = float(
                    getattr(
                        ds,
                        "RescaleSlope",
                        1.0
                    )
                )

                intercept = float(
                    getattr(
                        ds,
                        "RescaleIntercept",
                        0.0
                    )
                )

                hu = (
                    stored_pixels
                    * slope
                    + intercept
                )

                volume_slices.append(
                    hu
                )

        except Exception as error:

            print(
                "\nERROR reading CT pixel data:"
            )

            print(error)

            failed += 1

            reconstruction_reports.append({

                "patient_id": patient_id,

                "status":
                    "PIXEL_DATA_READ_ERROR",

                "error":
                    str(error)

            })

            continue

        # ----------------------------------------------------
        # Stack into 3-D volume
        # ----------------------------------------------------

        volume = np.stack(
            volume_slices,
            axis=2
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Create NIfTI affine
        # ----------------------------------------------------

        first_position = np.asarray(
            [
                float(x)
                for x in (
                    slice_entries[0]["ds"]
                    .ImagePositionPatient
                )
            ],
            dtype=np.float64
        )

        affine = np.eye(
            4,
            dtype=np.float64
        )

        affine[:3, 0] = (
            row_direction
            * row_spacing
        )

        affine[:3, 1] = (
            column_direction
            * column_spacing
        )

        affine[:3, 2] = (
            slice_direction
            * median_spacing
        )

        affine[:3, 3] = (
            first_position
        )

        # ----------------------------------------------------
        # Save NIfTI
        # ----------------------------------------------------

        patient_output_dir = (
            OUTPUT_ROOT
            / patient_id
        )

        patient_output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            patient_output_dir
            / f"{patient_id}_original_ct.nii.gz"
        )

        nifti_image = nib.Nifti1Image(
            volume,
            affine
        )

        nifti_image.set_data_dtype(
            np.float32
        )

        nib.save(
            nifti_image,
            str(output_file)
        )

        # ----------------------------------------------------
        # Reload and verify
        # ----------------------------------------------------

        loaded = nib.load(
            str(output_file)
        )

        loaded_shape = tuple(
            int(x)
            for x in loaded.shape
        )

        loaded_spacing = np.sqrt(
            np.sum(
                loaded.affine[:3, :3] ** 2,
                axis=0
            )
        )

        volume_min = float(
            np.min(volume)
        )

        volume_max = float(
            np.max(volume)
        )

        volume_mean = float(
            np.mean(volume)
        )

        # ----------------------------------------------------
        # Final validation
        # ----------------------------------------------------

        shape_ok = (
            loaded_shape
            == (
                rows,
                columns,
                expected_count
            )
        )

        spacing_ok = np.allclose(
            loaded_spacing,
            [
                row_spacing,
                column_spacing,
                median_spacing
            ],
            atol=1e-4
        )

        count_ok = (
            len(slice_entries)
            == expected_count
        )
        
        if (
            shape_ok
            and spacing_ok
            and count_ok
        ):

            status = (
                "CT_VOLUME_RECONSTRUCTION_SUCCESS"
            )

            successful += 1

        else:

            status = (
                "CT_VOLUME_RECONSTRUCTION_VALIDATION_FAILED"
            )

            failed += 1

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print(
            f"\nReferenced CT slices used: "
            f"{len(slice_entries)}"
        )

        print(
            f"Expected SEG references: "
            f"{expected_count}"
        )

        print(
            f"Image size: "
            f"{rows} x {columns}"
        )

        print(
            f"Reconstructed volume: "
            f"{volume.shape}"
        )

        print(
            f"Pixel spacing: "
            f"{row_spacing:.6f} x "
            f"{column_spacing:.6f} mm"
        )

        print(
            f"Slice spacing: "
            f"{median_spacing:.6f} mm"
        )

        print(
            f"HU range: "
            f"{volume_min:.2f} "
            f"to "
            f"{volume_max:.2f}"
        )

        print(
            f"Mean HU: "
            f"{volume_mean:.2f}"
        )

        print(
            "\nOutput:"
        )

        print(
            output_file
        )

        print(
            f"\nStatus: {status}"
        )

        # ----------------------------------------------------
        # Store patient report
        # ----------------------------------------------------

        reconstruction_reports.append({

            "patient_id": patient_id,

            "seg_file": str(
                seg_file
            ),

            "series_instance_uid":
                series_uid,

            "expected_referenced_ct_count":
                expected_count,

            "located_referenced_ct_count":
                len(referenced_ct),

            "missing_referenced_ct_count":
                len(missing_uids),

            "geometry": {

                "rows":
                    rows,

                "columns":
                    columns,

                "pixel_spacing_mm": [
                    row_spacing,
                    column_spacing
                ],

                "slice_spacing_mm":
                    median_spacing,

                "slice_spacing_min_mm":
                    min_spacing,

                "slice_spacing_max_mm":
                    max_spacing,

                "image_orientation_patient": [
                    float(x)
                    for x in reference_iop
                ],

                "slice_direction": [
                    float(x)
                    for x in slice_direction
                ],

                "first_image_position_patient": [
                    float(x)
                    for x in first_position
                ]
            },

            "volume": {

                "shape": [
                    int(x)
                    for x in volume.shape
                ],

                "dtype":
                    str(volume.dtype),

                "min_hu":
                    volume_min,

                "max_hu":
                    volume_max,

                "mean_hu":
                    volume_mean
            },

            "validation": {

                "referenced_slice_count_match":
                    bool(count_ok),

                "shape_match":
                    bool(shape_ok),

                "spacing_match":
                    bool(spacing_ok)

            },

            "output_file":
                str(output_file),

            "status":
                status

        })

    # --------------------------------------------------------
    # Save complete report
    # --------------------------------------------------------

    final_report = {

        "project":
            "AI-Based Adrenal Tumor Detection "
            "and Segmentation System",

        "step":
            "04",

        "description":
            "Reconstruction of original CT volumes "
            "using only CT SOP Instance UIDs explicitly "
            "referenced by the original TCIA DICOM SEG.",

        "patients_inspected":
            len(patient_reports),

        "successful":
            successful,

        "failed":
            failed,

        "reports":
            reconstruction_reports
    }

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_report,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "STEP 04 COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nPatients inspected: "
        f"{len(patient_reports)}"
    )

    print(
        f"Successful reconstructions: "
        f"{successful}"
    )

    print(
        f"Failed reconstructions: "
        f"{failed}"
    )

    print(
        "\nReport:"
    )

    print(
        REPORT_FILE
    )

    if (
        len(patient_reports) == 53
        and successful == 53
        and failed == 0
    ):

        print(
            "\nSTEP 04 STATUS: "
            "ALL 53 CT VOLUMES RECONSTRUCTED SUCCESSFULLY"
        )

    else:

        print(
            "\nSTEP 04 STATUS: "
            "REVIEW REQUIRED"
        )


if __name__ == "__main__":
    main()