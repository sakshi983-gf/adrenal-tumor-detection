from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent

STEP04_REPORT = (
    PROJECT_ROOT
    / "dataset" / "metadata" / "step04"
    / "ct_reconstruction_report.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset" / "metadata"
    / "patient_spacing.json"
)


print("=" * 70)
print("BUILD PATIENT SPACING LOOKUP")
print("=" * 70)

if not STEP04_REPORT.exists():
    print(f"\nERROR: Step 04 report not found at:")
    print(STEP04_REPORT)
    print("\nMake sure you have run 04_reconstruct_ct_volumes.py first.")
    raise SystemExit(1)


with open(STEP04_REPORT, "r") as f:
    data = json.load(f)


lookup = {}

for report in data.get("reports", []):
    pid = report.get("patient_id")
    geometry = report.get("geometry", {})

    pixel_spacing = geometry.get("pixel_spacing_mm", [None, None])
    slice_spacing = geometry.get("slice_spacing_mm", None)

    if pid and pixel_spacing[0] and pixel_spacing[1] and slice_spacing:
        lookup[pid] = {
            "row_mm": float(pixel_spacing[0]),
            "col_mm": float(pixel_spacing[1]),
            "slice_mm": float(slice_spacing)
        }


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(lookup, f, indent=2)


print(f"\nPatients with spacing: {len(lookup)}")
print(f"Output: {OUTPUT_FILE}")
print()

for i, (k, v) in enumerate(lookup.items()):
    if i >= 5:
        break
    print(f"  {k}: {v['row_mm']:.3f} x {v['col_mm']:.3f} x {v['slice_mm']:.3f} mm")

print("\nDONE.")
print("=" * 70)