from pathlib import Path
import os
import json
import random
import numpy as np
import nibabel as nib
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CT_ROOT = os.path.join(PROJECT_ROOT , "dataset", "ct_volumes")
MASK_ROOT = os.path.join(PROJECT_ROOT , "dataset", "tumor_ground_truth")
OUT_ROOT = os.path.join(PROJECT_ROOT , "dataset", "unet_dataset")

REPORT_DIR = os.path.join(PROJECT_ROOT , "dataset", "metadata", "step07")
REPORT_FILE = os.path.join(
    REPORT_DIR, "unet_dataset_report.json"
)

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

WINDOW_CENTER = 40
WINDOW_WIDTH = 400

LOW_HU = WINDOW_CENTER - WINDOW_WIDTH / 2
HIGH_HU = WINDOW_CENTER + WINDOW_WIDTH / 2

BACKGROUND_MARGIN = 2

os.makedirs(REPORT_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    os.makedirs(
        os.path.join(OUT_ROOT, split, "images"),
        exist_ok=True
    )
    os.makedirs(
        os.path.join(OUT_ROOT, split, "masks"),
        exist_ok=True
    )


# ============================================================
# PATIENTS
# ============================================================

patients = sorted(
    p for p in os.listdir(CT_ROOT)
    if p.startswith("Adrenal_Ki67_Seg_")
    and os.path.isdir(os.path.join(CT_ROOT, p))
)

if len(patients) != 53:
    raise RuntimeError(
        f"Expected 53 patients, found {len(patients)}"
    )


# ============================================================
# PATIENT-LEVEL SPLIT
# ============================================================

random.seed(SEED)

shuffled = patients.copy()
random.shuffle(shuffled)

train_count = int(len(shuffled) * TRAIN_RATIO)
val_count = int(len(shuffled) * VAL_RATIO)

splits = {
    "train": sorted(
        shuffled[:train_count]
    ),
    "val": sorted(
        shuffled[train_count:train_count + val_count]
    ),
    "test": sorted(
        shuffled[train_count + val_count:]
    )
}


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 07 - U-NET DATASET PREPARATION")
print("=" * 70)

print()
print("Total patients :", len(patients))
print("Train patients :", len(splits["train"]))
print("Val patients   :", len(splits["val"]))
print("Test patients  :", len(splits["test"]))

print()
print("CT window:")
print("  Low HU  :", LOW_HU)
print("  High HU :", HIGH_HU)
print()


# ============================================================
# REPORT
# ============================================================

report = {
    "seed": SEED,
    "ct_window": {
        "center": WINDOW_CENTER,
        "width": WINDOW_WIDTH,
        "low_hu": LOW_HU,
        "high_hu": HIGH_HU
    },
    "background_margin": BACKGROUND_MARGIN,
    "patient_split": splits,
    "splits": {}
}

total_images = 0
total_tumor = 0
total_background = 0


# ============================================================
# PROCESS SPLITS
# ============================================================

for split, split_patients in splits.items():

    print("=" * 70)
    print(split.upper())
    print("=" * 70)

    split_images = 0
    split_tumor = 0
    split_background = 0

    report["splits"][split] = {
        "patients": split_patients,
        "patients_count": len(split_patients)
    }

    for patient in split_patients:

        print(" ", patient)

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
            raise RuntimeError(
                f"CT missing: {patient}"
            )

        if not os.path.isfile(mask_file):
            raise RuntimeError(
                f"Mask missing: {patient}"
            )

        # ----------------------------------------------------
        # LOAD
        # ----------------------------------------------------

        ct = nib.load(ct_file)
        mask = nib.load(mask_file)

        ct_data = ct.get_fdata(
            dtype=np.float32
        )

        # IMPORTANT:
        # get_fdata requires floating point dtype
        mask_data = mask.get_fdata(
            dtype=np.float32
        )

        if ct_data.shape != mask_data.shape:
            raise RuntimeError(
                f"Shape mismatch: {patient}"
            )

        # ----------------------------------------------------
        # FIND TUMOR SLICES
        # ----------------------------------------------------

        tumor_slices = np.where(
            np.any(
                mask_data > 0,
                axis=(0, 1)
            )
        )[0]

        if len(tumor_slices) == 0:
            raise RuntimeError(
                f"Empty tumor mask: {patient}"
            )

        # ----------------------------------------------------
        # SELECT TUMOR + NEIGHBOR SLICES
        # ----------------------------------------------------

        selected = set()

        for z in tumor_slices:

            for offset in range(
                -BACKGROUND_MARGIN,
                BACKGROUND_MARGIN + 1
            ):

                zz = int(z + offset)

                if 0 <= zz < ct_data.shape[2]:
                    selected.add(zz)

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        image_dir = os.path.join(
            OUT_ROOT, split, "images"
        )

        mask_dir = os.path.join(
            OUT_ROOT, split, "masks"
        )

        for z in sorted(selected):

            ct_slice = ct_data[:, :, z]

            mask_slice = (
                mask_data[:, :, z] > 0
            ).astype(np.uint8)

            # CT window
            ct_slice = np.clip(
                ct_slice,
                LOW_HU,
                HIGH_HU
            )

            # Normalize 0-1 then save 8-bit PNG
            ct_slice = (
                (ct_slice - LOW_HU)
                / (HIGH_HU - LOW_HU)
                * 255.0
            ).astype(np.uint8)

            filename = (
                f"{patient}_slice_{z:03d}.png"
            )

            Image.fromarray(
                ct_slice
            ).save(
                os.path.join(
                    image_dir,
                    filename
                )
            )

            Image.fromarray(
                mask_slice * 255
            ).save(
                os.path.join(
                    mask_dir,
                    filename
                )
            )

            split_images += 1
            total_images += 1

            if np.any(mask_slice):
                split_tumor += 1
                total_tumor += 1
            else:
                split_background += 1
                total_background += 1

        del ct_data
        del mask_data

    report["splits"][split]["images"] = split_images
    report["splits"][split]["tumor_slices"] = split_tumor
    report["splits"][split]["background_slices"] = split_background

    print()
    print("Images       :", split_images)
    print("Tumor slices :", split_tumor)
    print("Background   :", split_background)
    print()


# ============================================================
# TOTALS
# ============================================================

report["totals"] = {
    "patients": len(patients),
    "images": total_images,
    "tumor_slices": total_tumor,
    "background_slices": total_background
}


# ============================================================
# SAVE REPORT
# ============================================================

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# FINAL
# ============================================================

print("=" * 70)
print("STEP 07 COMPLETE")
print("=" * 70)

print("Total patients    :", len(patients))
print("Train patients    :", len(splits["train"]))
print("Validation        :", len(splits["val"]))
print("Test patients     :", len(splits["test"]))
print("Total images      :", total_images)
print("Tumor slices      :", total_tumor)
print("Background slices :", total_background)

print()
print("Dataset :", OUT_ROOT)
print("Report  :", REPORT_FILE)

print()
print("STEP 07 STATUS: U-NET DATASET PREPARED")