from pathlib import Path
import os
import json
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MASK_ROOT = PROJECT_ROOT / "dataset" / "tumor_ground_truth"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "metadata" / "step19_growth"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "tumor_growth_report.json"


# ============================================================
# PATIENTS
# ============================================================

patients = sorted(
    p for p in os.listdir(MASK_ROOT)
    if p.startswith("Adrenal_Ki67_Seg_")
    and os.path.isdir(MASK_ROOT / p)
)

print("=" * 70)
print("TUMOR GROWTH ANALYSIS")
print("=" * 70)
print(f"Patients: {len(patients)}")
print()


# ============================================================
# PER-PATIENT ANALYSIS
# ============================================================

report = []

for n, patient in enumerate(patients, 1):

    mask_path = MASK_ROOT / patient / f"{patient}_tumor_mask.nii.gz"

    if not mask_path.exists():
        print(f"[{n:02d}] {patient} — mask not found")
        continue

    mask_img = nib.load(str(mask_path))
    mask_data = mask_img.get_fdata(dtype=np.float32)

    spacing = mask_img.header.get_zooms()[:3]
    pixel_area_mm2 = float(spacing[0]) * float(spacing[1])
    slice_thickness_mm = float(spacing[2])

    area_per_slice = np.sum(mask_data > 0, axis=(0, 1)) * pixel_area_mm2

    total_voxels = int(np.sum(mask_data > 0))
    volume_mm3 = float(total_voxels) * pixel_area_mm2 * slice_thickness_mm
    volume_cm3 = volume_mm3 / 1000.0

    tumor_slice_indices = np.where(area_per_slice > 0)[0]

    if len(tumor_slice_indices) == 0:
        continue

    first_slice = int(tumor_slice_indices[0])
    last_slice = int(tumor_slice_indices[-1])
    tumor_profile = area_per_slice[first_slice:last_slice + 1]

    peak_area_mm2 = float(np.max(tumor_profile))
    peak_slice = int(first_slice + np.argmax(tumor_profile))

    # ============================================================
    # PLOT — individual patient profile
    # ============================================================

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(first_slice, last_slice + 1)

    ax.plot(x, tumor_profile, color='#C0392B', linewidth=2)
    ax.fill_between(x, tumor_profile, alpha=0.2, color='#C0392B')

    ax.set_xlabel("CT Slice Index", fontsize=11)
    ax.set_ylabel("Tumor Cross-Section Area (mm²)", fontsize=11)
    ax.set_title(
        f"{patient} — Tumor Size Profile\n"
        f"Volume = {volume_cm3:.2f} cm³  |  Peak area = {peak_area_mm2:.1f} mm²",
        fontsize=11, fontweight='bold'
    )

    ax.grid(True, alpha=0.3)
    ax.axhline(peak_area_mm2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.annotate(
        f"Peak\n({peak_slice})",
        xy=(peak_slice, peak_area_mm2),
        xytext=(peak_slice + 2, peak_area_mm2 * 0.85),
        fontsize=8, color='#333',
        arrowprops=dict(arrowstyle='->', color='#555', lw=1)
    )

    plt.tight_layout()

    plot_file = OUTPUT_DIR / f"{patient}_tumor_profile.png"
    plt.savefig(plot_file, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    report.append({
        "patient": patient,
        "tumor_volume_cm3": round(float(volume_cm3), 3),
        "peak_area_mm2": round(float(peak_area_mm2), 2),
        "peak_slice": int(peak_slice),
        "first_slice": int(first_slice),
        "last_slice": int(last_slice),
        "num_tumor_slices": int(len(tumor_slice_indices)),
        "plot_file": str(plot_file)
    })

    print(f"[{n:02d}] {patient} — Volume {volume_cm3:.2f} cm³, "
          f"Peak {peak_area_mm2:.1f} mm² at slice {peak_slice}")


# ============================================================
# COMPARISON BAR CHART — all patients
# ============================================================

volumes = [r["tumor_volume_cm3"] for r in report]
names = [r["patient"].replace("Adrenal_Ki67_Seg_", "P") for r in report]

fig, ax = plt.subplots(figsize=(14, 5))

median_volume = float(np.median(volumes))

colors = ['#C0392B' if v > median_volume else '#2E86C1' for v in volumes]

ax.bar(names, volumes, color=colors, edgecolor='black', linewidth=0.6)

ax.set_xlabel("Patient ID", fontsize=11)
ax.set_ylabel("Tumor Volume (cm³)", fontsize=11)
ax.set_title(
    "Tumor Volume Comparison Across All Patients",
    fontsize=12, fontweight='bold'
)

ax.grid(True, alpha=0.3, axis='y')
ax.axhline(median_volume, color='green', linestyle='--', linewidth=1.4,
           label=f'Median = {median_volume:.1f} cm³')

ax.legend(fontsize=10)

plt.xticks(rotation=90, fontsize=8)
plt.tight_layout()

comparison_file = OUTPUT_DIR / "all_patients_tumor_volume_comparison.png"
plt.savefig(comparison_file, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Patients analyzed    : {len(report)}")
print(f"Mean tumor volume    : {float(np.mean(volumes)):.2f} cm³")
print(f"Median tumor volume  : {median_volume:.2f} cm³")
print(f"Min tumor volume     : {float(np.min(volumes)):.2f} cm³")
print(f"Max tumor volume     : {float(np.max(volumes)):.2f} cm³")
print()
print(f"Per-patient plots    : {OUTPUT_DIR}")
print(f"Comparison chart     : {comparison_file}")


# ============================================================
# SAVE REPORT
# ============================================================

with open(REPORT_FILE, "w") as f:
    json.dump(report, f, indent=2)

print(f"JSON report          : {REPORT_FILE}")
print("=" * 70)