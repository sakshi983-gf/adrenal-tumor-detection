from pathlib import Path
from collections import defaultdict
import pydicom


# ============================================================
# STEP 02
# INSPECT TCIA DICOM STRUCTURE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TCIA_ROOT = (
    PROJECT_ROOT
    / "raw"
    / "adrenal_acc_ki67_seg"
)


print("=" * 70)
print("STEP 02 - INSPECT TCIA DICOM STRUCTURE")
print("=" * 70)

print()
print("TCIA dataset:")
print(TCIA_ROOT)


if not TCIA_ROOT.exists():

    print()
    print("ERROR: TCIA dataset not found.")
    raise SystemExit(1)


# ============================================================
# DATA STRUCTURES
# ============================================================

patients = {}

total_dicom = 0
readable_dicom = 0


# ============================================================
# SCAN DATASET
# ============================================================

for patient_folder in sorted(
    [
        p for p in TCIA_ROOT.iterdir()
        if p.is_dir()
        and p.name.startswith("Adrenal_Ki67_Seg_")
    ]
):

    patient_id = patient_folder.name

    studies = defaultdict(
        lambda: defaultdict(list)
    )

    for path in patient_folder.rglob("*"):

        if not path.is_file():
            continue

        total_dicom += 1

        try:

            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True
            )

        except Exception:

            continue

        readable_dicom += 1

        modality = str(
            getattr(
                ds,
                "Modality",
                "UNKNOWN"
            )
        ).upper()

        study_uid = str(
            getattr(
                ds,
                "StudyInstanceUID",
                "UNKNOWN_STUDY"
            )
        )

        series_uid = str(
            getattr(
                ds,
                "SeriesInstanceUID",
                "UNKNOWN_SERIES"
            )
        )

        studies[
            study_uid
        ][
            series_uid
        ].append(
            {
                "path": str(path),
                "modality": modality,
                "rows": getattr(
                    ds,
                    "Rows",
                    None
                ),
                "columns": getattr(
                    ds,
                    "Columns",
                    None
                ),
                "pixel_spacing": getattr(
                    ds,
                    "PixelSpacing",
                    None
                ),
                "series_description": str(
                    getattr(
                        ds,
                        "SeriesDescription",
                        ""
                    )
                )
            }
        )

    patients[patient_id] = studies


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("-" * 70)
print("PATIENT / SERIES SUMMARY")
print("-" * 70)


for patient_id, studies in patients.items():

    print()
    print(patient_id)

    for study_uid, series_dict in studies.items():

        print(
            f"  Study: {study_uid}"
        )

        for series_uid, objects in series_dict.items():

            modality_counts = defaultdict(int)

            for obj in objects:

                modality_counts[
                    obj["modality"]
                ] += 1

            first = objects[0]

            print()
            print(
                f"    Series: {series_uid}"
            )

            print(
                f"      Objects: {len(objects)}"
            )

            print(
                f"      Modalities: "
                f"{dict(modality_counts)}"
            )

            print(
                f"      Rows: "
                f"{first['rows']}"
            )

            print(
                f"      Columns: "
                f"{first['columns']}"
            )

            print(
                f"      PixelSpacing: "
                f"{first['pixel_spacing']}"
            )

            print(
                f"      Description: "
                f"{first['series_description']}"
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("STEP 02 SUMMARY")
print("=" * 70)

print()
print(
    f"Patients inspected: "
    f"{len(patients)}"
)

print(
    f"Files scanned: "
    f"{total_dicom}"
)

print(
    f"Readable DICOM objects: "
    f"{readable_dicom}"
)

total_series = 0
total_ct_series = 0
total_seg_series = 0

for studies in patients.values():

    for series_dict in studies.values():

        total_series += len(series_dict)

        for objects in series_dict.values():

            modalities = {
                obj["modality"]
                for obj in objects
            }

            if "CT" in modalities:

                total_ct_series += 1

            if "SEG" in modalities:

                total_seg_series += 1


print(
    f"Total DICOM series: "
    f"{total_series}"
)

print(
    f"CT series: "
    f"{total_ct_series}"
)

print(
    f"SEG series: "
    f"{total_seg_series}"
)

print()
print("=" * 70)
print(
    "STEP 02 STATUS: "
    "DICOM STRUCTURE INSPECTION COMPLETE"
)
print("=" * 70)