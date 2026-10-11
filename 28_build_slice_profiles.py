from pathlib import Path
import os
import json
import numpy as np
import nibabel as nib


PROJECT_ROOT = Path(__file__).resolve().parent

MASK_ROOT = PROJECT_ROOT / "dataset" / "tumor_ground_truth"
SPACING_FILE = PROJECT_ROOT / "dataset" / "metadata" / "patient_spacing.json"

OUTPUT_FILE = PROJECT_ROOT / "dataset" / "metadata" / "slice_profiles.json"


print("=" * 70)
print("BUILD TUMOR PROFILES")
print("=" * 70)


with open(SPACING_FILE, "r") as f:
    spacing_lookup = json.load(f)

patients = sorted(
    p for p in os.listdir(MASK_ROOT)
    if p.startswith("Adrenal_Ki67_Seg_")
    and os.path.isdir(MASK_ROOT / p)
)

print(f"Processing {len(patients)} cases...\n")

profiles = {}

for n, patient in enumerate(patients, 1):

    mask_path = MASK_ROOT / patient / f"{patient}_tumor_mask.nii.gz"

    if not mask_path.exists():
        continue

    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)

    if patient in spacing_lookup:
        sp = spacing_lookup[patient]
        pixel_area_mm2 = sp["row_mm"] * sp["col_mm"]
    else:
        pixel_area_mm2 = 0.742 * 0.742

    area_per_slice_px = mask.sum(axis=(0, 1))
    area_per_slice_mm2 = area_per_slice_px * pixel_area_mm2

    tumor_slices = np.where(area_per_slice_px > 0)[0]

    if len(tumor_slices) == 0:
        continue

    peak_slice = int(tumor_slices[np.argmax(area_per_slice_px[tumor_slices])])
    peak_area = float(area_per_slice_mm2[peak_slice])

    profiles[patient] = {
        "total_slices": int(mask.shape[2]),
        "first_tumor_slice": int(tumor_slices[0]),
        "last_tumor_slice": int(tumor_slices[-1]),
        "peak_slice": peak_slice,
        "peak_area_mm2": round(peak_area, 2),
        "area_per_slice_mm2": [
            round(float(a), 2) for a in area_per_slice_mm2
        ]
    }

    if n % 10 == 0 or n == len(patients):
        print(f"  [{n:02d}/{len(patients)}] done")


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(profiles, f)

print()
print(f"Saved profiles for {len(profiles)} cases")
print(f"Output: {OUTPUT_FILE}")
print("=" * 70)