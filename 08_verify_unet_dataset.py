from pathlib import Path
from PIL import Image
import json
import re


# ============================================================
# STEP 08 - VERIFY U-NET DATASET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "dataset" / "unet_dataset"
REPORT_DIR = PROJECT_ROOT / "dataset" / "metadata" / "step08"
REPORT_FILE = REPORT_DIR / "unet_dataset_verification_report.json"

SPLITS = ["train", "val", "test"]


# ============================================================
# HELPER
# ============================================================

def extract_patient_id(filename):

    match = re.search(
        r"(Adrenal_Ki67_Seg_\d+)",
        filename
    )

    if match:
        return match.group(1)

    return None


def get_png_files(folder):

    return sorted(folder.glob("*.png"))


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 08 - U-NET DATASET VERIFICATION")
print("=" * 70)

if not DATASET_ROOT.exists():

    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATASET_ROOT}"
    )

print(f"Dataset found : {DATASET_ROOT}")
print()


# ============================================================
# REPORT
# ============================================================

report = {
    "dataset_root": str(DATASET_ROOT),
    "splits": {},
    "overall": {},
    "checks": {},
    "patient_leakage": {},
}


all_split_patients = {}

total_images = 0
total_masks = 0

overall_errors = []


# ============================================================
# CHECK EACH SPLIT
# ============================================================

for split in SPLITS:

    print("-" * 70)
    print(f"Checking {split.upper()} split")
    print("-" * 70)

    image_dir = DATASET_ROOT / split / "images"
    mask_dir = DATASET_ROOT / split / "masks"

    split_errors = []

    if not image_dir.exists():

        split_errors.append(
            f"Missing image folder: {image_dir}"
        )

    if not mask_dir.exists():

        split_errors.append(
            f"Missing mask folder: {mask_dir}"
        )

    if split_errors:

        print("ERROR")

        for error in split_errors:
            print(f"  {error}")

        overall_errors.extend(split_errors)

        report["splits"][split] = {
            "status": "FAIL",
            "errors": split_errors,
        }

        continue


    # --------------------------------------------------------
    # FILE LISTS
    # --------------------------------------------------------

    image_files = get_png_files(image_dir)
    mask_files = get_png_files(mask_dir)

    image_names = {
        file.name for file in image_files
    }

    mask_names = {
        file.name for file in mask_files
    }


    missing_masks = sorted(
        image_names - mask_names
    )

    missing_images = sorted(
        mask_names - image_names
    )


    if missing_masks:

        split_errors.append(
            f"{len(missing_masks)} image(s) have no mask"
        )


    if missing_images:

        split_errors.append(
            f"{len(missing_images)} mask(s) have no image"
        )


    paired_names = sorted(
        image_names & mask_names
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    dimension_errors = []
    mask_value_errors = []
    empty_mask_files = []

    patient_ids = set()
    patient_image_counts = {}


    for filename in paired_names:

        image_path = image_dir / filename
        mask_path = mask_dir / filename


        # ----------------------------------------------------
        # PATIENT ID
        # ----------------------------------------------------

        patient_id = extract_patient_id(filename)

        if patient_id is None:

            split_errors.append(
                f"Cannot determine patient ID: {filename}"
            )

        else:

            patient_ids.add(patient_id)

            patient_image_counts[patient_id] = (
                patient_image_counts.get(patient_id, 0) + 1
            )


        # ----------------------------------------------------
        # IMAGE / MASK
        # ----------------------------------------------------

        try:

            with Image.open(image_path) as image:

                image_size = image.size

            with Image.open(mask_path) as mask:

                mask_size = mask.size

                # Avoid deprecated Image.getdata()
                mask_values = set(
                    mask.convert("L").get_flattened_data()
                )


            # ------------------------------------------------
            # DIMENSIONS
            # ------------------------------------------------

            if image_size != mask_size:

                dimension_errors.append({
                    "file": filename,
                    "image_size": image_size,
                    "mask_size": mask_size,
                })


            # ------------------------------------------------
            # MASK VALUES
            # ------------------------------------------------

            invalid_values = sorted(
                mask_values - {0, 255}
            )

            if invalid_values:

                mask_value_errors.append({
                    "file": filename,
                    "invalid_values": invalid_values,
                })


            # ------------------------------------------------
            # EMPTY MASK
            # ------------------------------------------------

            if 255 not in mask_values:

                empty_mask_files.append(filename)


        except Exception as exc:

            split_errors.append(
                f"Error reading {filename}: {exc}"
            )


    # --------------------------------------------------------
    # STORE PATIENT INFORMATION
    # --------------------------------------------------------

    all_split_patients[split] = sorted(
        patient_ids
    )


    # --------------------------------------------------------
    # COUNTS
    # --------------------------------------------------------

    split_image_count = len(image_files)
    split_mask_count = len(mask_files)

    total_images += split_image_count
    total_masks += split_mask_count


    # --------------------------------------------------------
    # SPLIT STATUS
    # --------------------------------------------------------

    split_status = (

        "PASS"

        if (
            not split_errors
            and not missing_masks
            and not missing_images
            and not dimension_errors
            and not mask_value_errors
        )

        else "FAIL"
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(
        f"Patients             : {len(patient_ids)}"
    )

    print(
        f"Images               : {split_image_count}"
    )

    print(
        f"Masks                : {split_mask_count}"
    )

    print(
        f"Paired samples       : {len(paired_names)}"
    )

    print(
        f"Dimension errors     : {len(dimension_errors)}"
    )

    print(
        f"Invalid mask values  : {len(mask_value_errors)}"
    )

    print(
        f"Empty masks          : {len(empty_mask_files)}"
    )

    print(
        f"Status               : {split_status}"
    )

    print()


    report["splits"][split] = {

        "status": split_status,

        "patients": sorted(patient_ids),

        "patient_count": len(patient_ids),

        "image_count": split_image_count,

        "mask_count": split_mask_count,

        "paired_samples": len(paired_names),

        "missing_masks": missing_masks,

        "missing_images": missing_images,

        "dimension_errors": dimension_errors,

        "mask_value_errors": mask_value_errors,

        "empty_mask_count": len(empty_mask_files),

        "patient_image_counts":
            patient_image_counts,
    }


    overall_errors.extend(
        split_errors
    )


# ============================================================
# PATIENT LEAKAGE CHECK
# ============================================================

print("-" * 70)
print("Checking patient-level data leakage")
print("-" * 70)


train_patients = set(
    all_split_patients.get("train", [])
)

val_patients = set(
    all_split_patients.get("val", [])
)

test_patients = set(
    all_split_patients.get("test", [])
)


train_val_overlap = sorted(
    train_patients & val_patients
)

train_test_overlap = sorted(
    train_patients & test_patients
)

val_test_overlap = sorted(
    val_patients & test_patients
)


leakage_found = (

    bool(train_val_overlap)
    or bool(train_test_overlap)
    or bool(val_test_overlap)
)


print(
    f"Train / Val overlap  : "
    f"{len(train_val_overlap)}"
)

print(
    f"Train / Test overlap : "
    f"{len(train_test_overlap)}"
)

print(
    f"Val / Test overlap   : "
    f"{len(val_test_overlap)}"
)


if leakage_found:

    print("PATIENT LEAKAGE: FAIL")

else:

    print("PATIENT LEAKAGE: PASS")


print()


# ============================================================
# TOTAL PATIENTS
# ============================================================

all_patients = (
    train_patients
    | val_patients
    | test_patients
)


print("-" * 70)
print("Overall dataset statistics")
print("-" * 70)

print(
    f"Total unique patients : "
    f"{len(all_patients)}"
)

print(
    f"Total images          : "
    f"{total_images}"
)

print(
    f"Total masks           : "
    f"{total_masks}"
)

print()


# ============================================================
# FINAL CHECKS
# ============================================================

expected_patients = 53
expected_images = 2095


checks = {

    "dataset_exists":
        DATASET_ROOT.exists(),

    "expected_patient_count":
        len(all_patients) == expected_patients,

    "image_mask_count_match":
        total_images == total_masks,

    "expected_image_count":
        total_images == expected_images,

    "patient_leakage_free":
        not leakage_found,

    "train_split_present":
        len(train_patients) > 0,

    "val_split_present":
        len(val_patients) > 0,

    "test_split_present":
        len(test_patients) > 0,

    "all_split_checks_pass":
        all(
            report["splits"].get(split, {}).get(
                "status"
            ) == "PASS"
            for split in SPLITS
        ),
}


overall_pass = all(
    checks.values()
)


# ============================================================
# REPORT
# ============================================================

report["overall"] = {

    "unique_patients":
        len(all_patients),

    "total_images":
        total_images,

    "total_masks":
        total_masks,

    "expected_patients":
        expected_patients,

    "expected_images":
        expected_images,
}


report["checks"] = checks


report["patient_leakage"] = {

    "train_val_overlap":
        train_val_overlap,

    "train_test_overlap":
        train_test_overlap,

    "val_test_overlap":
        val_test_overlap,

    "leakage_free":
        not leakage_found,
}


report["status"] = (
    "PASS"
    if overall_pass
    else "FAIL"
)


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("=" * 70)
print("STEP 08 COMPLETE")
print("=" * 70)

print(
    f"Unique patients : "
    f"{len(all_patients)}"
)

print(
    f"Total images    : "
    f"{total_images}"
)

print(
    f"Total masks     : "
    f"{total_masks}"
)

print()


for split in SPLITS:

    split_report = report["splits"].get(
        split,
        {}
    )

    print(
        f"{split.capitalize():12} : "
        f"{split_report.get('status', 'FAIL')}"
    )


print()

print(
    f"Patient leakage : "
    f"{'PASS' if not leakage_found else 'FAIL'}"
)

print()

print(
    f"Report : {REPORT_FILE}"
)

print()


if overall_pass:

    print(
        "STEP 08 STATUS: "
        "U-NET DATASET VERIFIED SUCCESSFULLY"
    )

else:

    print(
        "STEP 08 STATUS: "
        "VERIFICATION FAILED"
    )

    print()
    print("Review the report for errors.")