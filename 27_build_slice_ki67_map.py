from pathlib import Path
import os
import json
import numpy as np
import nibabel as nib


PROJECT_ROOT = Path(__file__).resolve().parent

MASK_ROOT = PROJECT_ROOT / "dataset" / "tumor_ground_truth"
CLINICAL_FILE = PROJECT_ROOT / "dataset" / "metadata" / "patient_clinical.json"
KI67_MODEL_PREDS = PROJECT_ROOT / "dataset" / "metadata" / "ki67_predictions.json"

OUTPUT_FILE = PROJECT_ROOT / "dataset" / "metadata" / "slice_ki67_map.json"


print("=" * 70)
print("BUILD SLICE-LEVEL KI-67 MAP")
print("=" * 70)


with open(CLINICAL_FILE, "r") as f:
    clinical = json.load(f)

model_preds = {}
if KI67_MODEL_PREDS.exists():
    with open(KI67_MODEL_PREDS, "r") as f:
        model_preds = json.load(f)
    print(f"Using AI-predicted Ki-67 for patients with model output")


patients = sorted(
    p for p in os.listdir(MASK_ROOT)
    if p.startswith("Adrenal_Ki67_Seg_")
    and os.path.isdir(MASK_ROOT / p)
)

slice_ki67_map = {}

for patient in patients:

    mask_path = MASK_ROOT / patient / f"{patient}_tumor_mask.nii.gz"
    if not mask_path.exists():
        continue

    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
    area_per_slice = mask.sum(axis=(0, 1))
    tumor_slices = np.where(area_per_slice > 0)[0]

    if len(tumor_slices) == 0:
        continue

    # Get patient-level Ki-67 (prefer AI-predicted, fall back to clinical)
    if patient in model_preds:
        patient_ki67 = model_preds[patient]["predicted_ki67"]
        source = "ai_predicted"
    elif patient in clinical and clinical[patient]["ki67"] is not None:
        patient_ki67 = clinical[patient]["ki67"]
        source = "clinical"
    else:
        continue

    peak_area = float(area_per_slice[tumor_slices].max())
    if peak_area <= 0:
        continue

    # Peak slice
    peak_slice = int(tumor_slices[np.argmax(area_per_slice[tumor_slices])])

    # Compute per-slice estimates
    slice_estimates = {}
    for s in tumor_slices:
        s = int(s)
        slice_area = float(area_per_slice[s])
        ratio = slice_area / peak_area
        # Power of 0.75 gives a smooth curve (not too extreme at edges)
        weight = ratio ** 0.75
        estimated = patient_ki67 * weight
        slice_estimates[str(s)] = {
            "slice_area_px": int(slice_area),
            "weight": round(weight, 4),
            "estimated_ki67": round(estimated, 2)
        }

    slice_ki67_map[patient] = {
        "patient_ki67": round(float(patient_ki67), 2),
        "ki67_source": source,
        "peak_slice": peak_slice,
        "peak_area_px": int(peak_area),
        "first_slice": int(tumor_slices[0]),
        "last_slice": int(tumor_slices[-1]),
        "num_tumor_slices": int(len(tumor_slices)),
        "slice_estimates": slice_estimates
    }


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(slice_ki67_map, f, indent=2)

print()
print(f"Patients processed: {len(slice_ki67_map)}")
print(f"Output: {OUTPUT_FILE}")
print()

# Show a sample
for i, (pid, data) in enumerate(slice_ki67_map.items()):
    if i >= 2:
        break
    print(f"\n{pid}: patient Ki-67 = {data['patient_ki67']}% ({data['ki67_source']})")
    print(f"  Peak slice: {data['peak_slice']}  |  {data['num_tumor_slices']} tumor slices")
    print(f"  Sample slice values:")
    for s in list(data['slice_estimates'].keys())[:5]:
        v = data['slice_estimates'][s]
        print(f"    slice {s}: est Ki-67 = {v['estimated_ki67']}%")

print()
print("DONE.")
print("=" * 70)