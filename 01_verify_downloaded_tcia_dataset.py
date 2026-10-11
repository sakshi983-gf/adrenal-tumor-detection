
# ============================================================
# STEP 01
# VERIFY DOWNLOADED TCIA DATASET
# ============================================================

from pathlib import Path
import pydicom

PROJECT_ROOT = Path(__file__).resolve().parent

TCIA_ROOT = (
    PROJECT_ROOT
    / "raw"
    / "adrenal_acc_ki67_seg"
)


print("=" * 70)
print("STEP 01 - VERIFY DOWNLOADED TCIA DATASET")
print("=" * 70)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("TCIA dataset:")
print(TCIA_ROOT)


# ------------------------------------------------------------
# Check dataset folder
# ------------------------------------------------------------

if not TCIA_ROOT.exists():

    print()
    print("ERROR: TCIA dataset folder not found.")
    print(TCIA_ROOT)

    raise SystemExit(1)

print()
print("TCIA dataset folder: FOUND")


# ------------------------------------------------------------
# Find patient folders
# ------------------------------------------------------------

patient_folders = sorted(
    [
        path
        for path in TCIA_ROOT.iterdir()
        if path.is_dir()
    ]
)


print()
print(
    f"Patient folders found: "
    f"{len(patient_folders)}"
)


# ------------------------------------------------------------
# Scan DICOM files
# ------------------------------------------------------------

dicom_count = 0
readable_dicom_count = 0

modality_counts = {}

patient_counts = {}


for patient_folder in patient_folders:

    patient_dicom_count = 0

    for path in patient_folder.rglob("*"):

        if not path.is_file():
            continue

        dicom_count += 1

        try:

            ds = pydicom.dcmread(
                str(path),
                stop_before_pixels=True,
                force=True
            )

        except Exception:

            continue

        readable_dicom_count += 1

        modality = str(
            getattr(
                ds,
                "Modality",
                "UNKNOWN"
            )
        ).upper()

        modality_counts[modality] = (
            modality_counts.get(
                modality,
                0
            ) + 1
        )

        patient_dicom_count += 1

    patient_counts[
        patient_folder.name
    ] = patient_dicom_count


# ------------------------------------------------------------
# Print patient information
# ------------------------------------------------------------

print()
print("-" * 70)
print("PATIENT DATA")
print("-" * 70)

for patient, count in patient_counts.items():

    print(
        f"{patient}: "
        f"{count} readable DICOM files"
    )

# ------------------------------------------------------------
# Print modality information
# ------------------------------------------------------------

print()
print("-" * 70)
print("DICOM MODALITIES")
print("-" * 70)

for modality, count in sorted(
    modality_counts.items()
):

    print(
        f"{modality}: {count}"
    )

# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

print()
print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print()
print(
    f"Patient folders: "
    f"{len(patient_folders)}"
)

print(
    f"Files scanned: "
    f"{dicom_count}"
)

print(
    f"Readable DICOM objects: "
    f"{readable_dicom_count}"
)

print()
print("=" * 70)

if (
    len(patient_folders) > 0
    and readable_dicom_count > 0
):

    print(
        "STEP 01 STATUS: "
        "TCIA DATASET VERIFIED"
    )

else:

    print(
        "STEP 01 STATUS: "
        "DATASET REQUIRES REVIEW"
    )

print("=" * 70)