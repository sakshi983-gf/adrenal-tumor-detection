from pathlib import Path
import json
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

# Update this path if your file has a different name
XLSX_FILE = (
    PROJECT_ROOT
    / "raw" / "adrenal_acc_ki67_seg"
    / "Adrenal-ACC-Ki67-Seg_SupportingData_20230522 (4).xlsx"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "dataset" / "metadata"
    / "patient_clinical.json"
)


print("=" * 70)
print("BUILD CLINICAL LOOKUP")
print("=" * 70)

if not XLSX_FILE.exists():
    print(f"\nERROR: Clinical file not found at:")
    print(XLSX_FILE)
    print("\nPlease update XLSX_FILE path in this script.")
    raise SystemExit(1)


# Read the first sheet
df = pd.read_excel(XLSX_FILE, sheet_name=0)

print(f"Rows loaded : {len(df)}")
print(f"Columns     : {list(df.columns)}")
print()


STAGE_LABELS = {
    1: "Stage I",
    2: "Stage II",
    3: "Stage III",
    4: "Stage IV",
}


def safe_str(v):
    if pd.isna(v):
        return "N/A"
    return str(v).strip()


def safe_float(v):
    try:
        if pd.isna(v):
            return None
        return float(v)
    except Exception:
        return None


def safe_int(v):
    try:
        if pd.isna(v):
            return None
        return int(float(v))
    except Exception:
        return None


lookup = {}

for _, row in df.iterrows():

    pid = str(row["Patient_ID"]).strip()

    stage_num = safe_int(row.get("TNM_Stage"))

    entry = {
        "age": safe_str(row.get("Age")),
        "sex": safe_str(row.get("SEX")),
        "race": safe_str(row.get("Race")),
        "laterality": safe_str(row.get("Laterality")),
        "path_tumor_size_cm": safe_float(row.get("PathTumorSize")),
        "stage_number": stage_num,
        "stage_label": STAGE_LABELS.get(stage_num, "Unknown"),
        "T_stage": safe_str(row.get("TStaging")),
        "N_stage": safe_str(row.get("NStaging")),
        "M_stage": safe_str(row.get("MStaging")),
        "ki67": safe_float(row.get("Ki67")),
        "clinic_presentation": safe_str(row.get("CLINIC PRESENT")),
        "mets_at_dx": safe_str(row.get("MetsAtDX")),
        "resection_margin": safe_str(row.get("ResectionMargin")),
        "days_to_diagnosis": safe_int(row.get("DaysToDiagnosis")),
    }

    lookup[pid] = entry


OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as f:
    json.dump(lookup, f, indent=2)


print(f"Patients in lookup : {len(lookup)}")
print(f"Output             : {OUTPUT_FILE}")
print()
print("Sample entries:")

for i, (k, v) in enumerate(lookup.items()):
    if i >= 5:
        break
    print(f"  {k}: {v['stage_label']} | "
          f"T{v['T_stage'][1:]} N{v['N_stage'][1:]} M{v['M_stage'][1:]} | "
          f"Ki-67 = {v['ki67']}")

print()
print("DONE.")
print("=" * 70)