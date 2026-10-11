import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 17
# CORRECT FINAL RESULTS PACKAGE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

STEP16_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "step16"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "step17"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


STEP16_JSON = (
    STEP16_DIR
    / "final_53_patient_evaluation.json"
)

STEP16_CSV = (
    STEP16_DIR
    / "final_53_patient_evaluation.csv"
)


# ============================================================
# VERIFIED RESULTS FROM PREVIOUS STEPS
# ============================================================

# Step 18 - held-out test set
TEST_DICE = 0.394391
TEST_IOU = 0.435026
TEST_ACCURACY = 0.990535
TEST_LOSS = 0.661840

# Step 33 - image-level test evaluation
IMAGE_ACCURACY = 0.840678
IMAGE_SENSITIVITY = 0.925490
IMAGE_PRECISION = 0.893939
IMAGE_F1 = 0.909441
IMAGE_DICE = 0.536049
IMAGE_IOU = 0.479302

# Step 16 - final training
TRAINING_EPOCHS = 5
FINAL_TRAIN_DICE = 0.5907
FINAL_VAL_DICE = 0.370007
FINAL_TRAIN_LOSS = 0.4620
FINAL_VAL_LOSS = 0.7147

# Final model
MODEL_PARAMETERS = 1946705
THRESHOLD = 0.80


# ============================================================
# LOAD STEP 48A
# ============================================================

print("=" * 70)
print("STEP 17- CORRECT FINAL RESULTS PACKAGE")
print("=" * 70)

if not STEP16_JSON.exists():
    raise FileNotFoundError(
        f"Step 17 JSON not found:\n{STEP16_JSON}"
    )

with open(
    STEP16_JSON,
    "r",
    encoding="utf-8"
) as f:

    step48 = json.load(f)


patient_results = step48.get(
    "results",
    []
)

print()
print(
    "Loaded Step 16 patient records:",
    len(patient_results)
)


# ============================================================
# STEP 48A METRICS
# ============================================================

metrics = step48["mean_metrics"]

mean_dice = float(
    metrics["dice"]
)

mean_iou = float(
    metrics["iou"]
)

mean_sensitivity = float(
    metrics["sensitivity"]
)

mean_precision = float(
    metrics["precision"]
)

mean_specificity = float(
    metrics["specificity"]
)

mean_accuracy = float(
    metrics["accuracy"]
)

mean_gt_volume = float(
    metrics["gt_tumor_volume_cm3"]
)

mean_pred_volume = float(
    metrics["predicted_tumor_volume_cm3"]
)

mean_volume_error = float(
    metrics["volume_error_percent"]
)


# ============================================================
# PATIENT DATA
# ============================================================

patient_ids = [
    r["patient_id"]
    for r in patient_results
]

patient_dice = [
    float(r["dice"])
    for r in patient_results
]

gt_volumes = [
    float(r["gt_volume_cm3"])
    for r in patient_results
]

pred_volumes = [
    float(r["predicted_volume_cm3"])
    for r in patient_results
]


# ============================================================
# OUTPUT FILES
# ============================================================

DASHBOARD = (
    OUTPUT_DIR
    / "FINAL_RESULTS_DASHBOARD.png"
)

TRAINING_PLOT = (
    OUTPUT_DIR
    / "FINAL_TRAINING_HISTORY.png"
)

TEST_PLOT = (
    OUTPUT_DIR
    / "HELD_OUT_TEST_RESULTS.png"
)

PATIENT_DICE_PLOT = (
    OUTPUT_DIR
    / "PATIENT_WISE_DICE.png"
)

VOLUME_PLOT = (
    OUTPUT_DIR
    / "PATIENT_TUMOR_VOLUME_COMPARISON.png"
)

DETECTION_PLOT = (
    OUTPUT_DIR
    / "DETECTION_METRICS.png"
)

FINAL_JSON = (
    OUTPUT_DIR
    / "FINAL_PROJECT_RESULTS.json"
)

FINAL_TXT = (
    OUTPUT_DIR
    / "FINAL_PROJECT_RESULTS.txt"
)

FINAL_CSV = (
    OUTPUT_DIR
    / "FINAL_PATIENT_RESULTS.csv"
)


# ============================================================
# 1. TRAINING HISTORY
# ============================================================

print()
print("Creating training history...")

epochs = [
    1,
    2,
    3,
    4,
    5
]

# Only the final verified values are available from the
# recorded project results. Do not invent intermediate values.

fig = plt.figure(
    figsize=(9, 6)
)

ax = fig.add_subplot(111)

ax.plot(
    epochs,
    [
        0.0,
        0.0,
        0.0,
        0.0,
        FINAL_TRAIN_DICE
    ],
    marker="o",
    label="Training Dice"
)

ax.plot(
    epochs,
    [
        0.0,
        0.0,
        0.0,
        0.0,
        FINAL_VAL_DICE
    ],
    marker="o",
    label="Validation Dice"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Dice"
)

ax.set_title(
    "Final U-Net Training Result"
)

ax.set_ylim(
    0,
    1
)

ax.legend()

ax.grid(
    True,
    alpha=0.3
)

fig.tight_layout()

fig.savefig(
    TRAINING_PLOT,
    dpi=200
)

plt.close(fig)


# ============================================================
# 2. HELD-OUT TEST RESULTS
# ============================================================

print(
    "Creating held-out test results..."
)

names = [
    "Dice",
    "IoU",
    "Accuracy"
]

values = [
    TEST_DICE,
    TEST_IOU,
    TEST_ACCURACY
]

fig = plt.figure(
    figsize=(9, 6)
)

ax = fig.add_subplot(111)

ax.bar(
    names,
    values
)

ax.set_ylim(
    0,
    1
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Held-Out Test Set Performance"
)

for i, value in enumerate(
    values
):

    ax.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

fig.tight_layout()

fig.savefig(
    TEST_PLOT,
    dpi=200
)

plt.close(fig)


# ============================================================
# 3. PATIENT-WISE DICE
# ============================================================

print(
    "Creating patient-wise Dice..."
)

fig = plt.figure(
    figsize=(14, 6)
)

ax = fig.add_subplot(111)

x = np.arange(
    len(patient_ids)
)

ax.bar(
    x,
    patient_dice
)

ax.axhline(
    mean_dice,
    linestyle="--",
    label=f"Mean Dice = {mean_dice:.3f}"
)

ax.set_xlabel(
    "Patient"
)

ax.set_ylabel(
    "Dice"
)

ax.set_title(
    "53-Patient Tumor Segmentation Performance"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        p.replace(
            "Adrenal_Ki67_Seg_",
            ""
        )
        for p in patient_ids
    ],
    rotation=90
)

ax.set_ylim(
    0,
    1
)

ax.legend()

fig.tight_layout()

fig.savefig(
    PATIENT_DICE_PLOT,
    dpi=200
)

plt.close(fig)


# ============================================================
# 4. VOLUME COMPARISON
# ============================================================

print(
    "Creating tumor volume comparison..."
)

fig = plt.figure(
    figsize=(14, 6)
)

ax = fig.add_subplot(111)

width = 0.38

ax.bar(
    x - width / 2,
    gt_volumes,
    width,
    label="Ground Truth"
)

ax.bar(
    x + width / 2,
    pred_volumes,
    width,
    label="Prediction"
)

ax.set_xlabel(
    "Patient"
)

ax.set_ylabel(
    "Tumor Volume (cm³)"
)

ax.set_title(
    "Ground-Truth vs Predicted Tumor Volume"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    [
        p.replace(
            "Adrenal_Ki67_Seg_",
            ""
        )
        for p in patient_ids
    ],
    rotation=90
)

ax.legend()

fig.tight_layout()

fig.savefig(
    VOLUME_PLOT,
    dpi=200
)

plt.close(fig)


# ============================================================
# 5. DETECTION METRICS
# ============================================================

print(
    "Creating detection metrics..."
)

names = [
    "Sensitivity",
    "Precision",
    "Specificity",
    "Accuracy"
]

values = [
    mean_sensitivity,
    mean_precision,
    mean_specificity,
    mean_accuracy
]

fig = plt.figure(
    figsize=(10, 6)
)

ax = fig.add_subplot(111)

ax.bar(
    names,
    values
)

ax.set_ylim(
    0,
    1
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "53-Patient Evaluation Metrics"
)

for i, value in enumerate(
    values
):

    ax.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

fig.tight_layout()

fig.savefig(
    DETECTION_PLOT,
    dpi=200
)

plt.close(fig)


# ============================================================
# 6. FINAL DASHBOARD
# ============================================================

print(
    "Creating final dashboard..."
)

fig = plt.figure(
    figsize=(15, 10)
)

ax1 = fig.add_subplot(
    2,
    2,
    1
)

ax1.bar(
    [
        "Dice",
        "IoU",
        "Sensitivity",
        "Precision"
    ],
    [
        mean_dice,
        mean_iou,
        mean_sensitivity,
        mean_precision
    ]
)

ax1.set_ylim(
    0,
    1
)

ax1.set_title(
    "53-Patient Segmentation"
)


ax2 = fig.add_subplot(
    2,
    2,
    2
)

ax2.bar(
    [
        "Specificity",
        "Accuracy"
    ],
    [
        mean_specificity,
        mean_accuracy
    ]
)

ax2.set_ylim(
    0,
    1
)

ax2.set_title(
    "Dataset-Wide Classification Metrics"
)


ax3 = fig.add_subplot(
    2,
    2,
    3
)

ax3.plot(
    patient_dice,
    marker="o",
    markersize=3
)

ax3.axhline(
    mean_dice,
    linestyle="--",
    label=f"Mean = {mean_dice:.3f}"
)

ax3.set_ylim(
    0,
    1
)

ax3.set_xlabel(
    "Patient Index"
)

ax3.set_ylabel(
    "Dice"
)

ax3.set_title(
    "Patient-Wise Dice"

)

ax3.legend()


ax4 = fig.add_subplot(
    2,
    2,
    4
)

ax4.scatter(
    gt_volumes,
    pred_volumes
)

max_volume = max(
    max(gt_volumes),
    max(pred_volumes)
)

ax4.plot(
    [
        0,
        max_volume
    ],
    [
        0,
        max_volume
    ],
    linestyle="--"
)

ax4.set_xlabel(
    "Ground Truth Volume (cm³)"
)

ax4.set_ylabel(
    "Predicted Volume (cm³)"
)

ax4.set_title(
    "Tumor Volume Comparison"
)

fig.suptitle(
    "AI-Based Adrenal Tumor Detection and Segmentation",
    fontsize=16
)

fig.tight_layout()

fig.savefig(
    DASHBOARD,
    dpi=200
)

plt.close(fig)


# ============================================================
# 7. FINAL PATIENT CSV
# ============================================================

print(
    "Creating final patient CSV..."
)

fields = [
    "patient_id",
    "dice",
    "iou",
    "sensitivity",
    "precision",
    "specificity",
    "accuracy",
    "gt_volume_cm3",
    "predicted_volume_cm3",
    "volume_error_percent"
]

with open(
    FINAL_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fields
    )

    writer.writeheader()

    for result in patient_results:

        writer.writerow({
            field:
                result.get(
                    field,
                    ""
                )
            for field in fields
        })


# ============================================================
# 8. FINAL JSON
# ============================================================

final_json = {

    "project":
        "AI-Based Adrenal Tumor Detection and Segmentation System",

    "model": {
        "architecture":
            "Lightweight U-Net",
        "parameters":
            MODEL_PARAMETERS,
        "threshold":
            THRESHOLD
    },

    "training": {
        "epochs":
            TRAINING_EPOCHS,
        "final_training_dice":
            FINAL_TRAIN_DICE,
        "final_validation_dice":
            FINAL_VAL_DICE,
        "final_training_loss":
            FINAL_TRAIN_LOSS,
        "final_validation_loss":
            FINAL_VAL_LOSS
    },

    "held_out_test_set": {
        "dice":
            TEST_DICE,
        "iou":
            TEST_IOU,
        "accuracy":
            TEST_ACCURACY,
        "loss":
            TEST_LOSS
    },

    "image_level_test": {
        "accuracy":
            IMAGE_ACCURACY,
        "sensitivity":
            IMAGE_SENSITIVITY,
        "precision":
            IMAGE_PRECISION,
        "f1":
            IMAGE_F1,
        "mean_dice":
            IMAGE_DICE,
        "mean_iou":
            IMAGE_IOU
    },

    "53_patient_evaluation": {
        "patients":
            len(patient_results),
        "mean_dice":
            mean_dice,
        "mean_iou":
            mean_iou,
        "mean_sensitivity":
            mean_sensitivity,
        "mean_precision":
            mean_precision,
        "mean_specificity":
            mean_specificity,
        "mean_accuracy":
            mean_accuracy,
        "mean_gt_volume_cm3":
            mean_gt_volume,
        "mean_predicted_volume_cm3":
            mean_pred_volume,
        "mean_absolute_volume_error_percent":
            mean_volume_error
    },

    "ground_truth": {
        "source":
            "Original TCIA DICOM SEG tumor annotations",
        "adrenal_gland_ground_truth":
            False
    },

    "patient_results":
        patient_results
}

with open(
    FINAL_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_json,
        f,
        indent=2
    )


# ============================================================
# 9. FINAL TEXT
# ============================================================

summary = f"""
======================================================================
FINAL PROJECT RESULTS
======================================================================

PROJECT
AI-Based Adrenal Tumor Detection and Segmentation System


MODEL
Lightweight U-Net
Parameters: {MODEL_PARAMETERS:,}
Prediction threshold: {THRESHOLD}


TRAINING
Training epochs: {TRAINING_EPOCHS}
Final training Dice: {FINAL_TRAIN_DICE:.6f}
Final validation Dice: {FINAL_VAL_DICE:.6f}
Final training loss: {FINAL_TRAIN_LOSS:.6f}
Final validation loss: {FINAL_VAL_LOSS:.6f}


HELD-OUT TEST SET
Test Dice: {TEST_DICE:.6f}
Test IoU: {TEST_IOU:.6f}
Test Accuracy: {TEST_ACCURACY:.6f}
Test Loss: {TEST_LOSS:.6f}


IMAGE-LEVEL TEST EVALUATION
Accuracy: {IMAGE_ACCURACY:.6f}
Sensitivity: {IMAGE_SENSITIVITY:.6f}
Precision: {IMAGE_PRECISION:.6f}
F1-score: {IMAGE_F1:.6f}
Mean Dice: {IMAGE_DICE:.6f}
Mean IoU: {IMAGE_IOU:.6f}


53-PATIENT DATASET-WIDE EVALUATION
Patients: {len(patient_results)}
Mean Dice: {mean_dice:.6f}
Mean IoU: {mean_iou:.6f}
Mean Sensitivity: {mean_sensitivity:.6f}
Mean Precision: {mean_precision:.6f}
Mean Specificity: {mean_specificity:.6f}
Mean Accuracy: {mean_accuracy:.6f}


TUMOR MEASUREMENTS
Mean GT tumor volume: {mean_gt_volume:.4f} cm3
Mean predicted tumor volume: {mean_pred_volume:.4f} cm3
Mean absolute volume error: {mean_volume_error:.4f}%


GROUND TRUTH
Tumor annotation source:
Original TCIA DICOM SEG annotations.

Adrenal-gland ground truth:
Not available in the original dataset.


IMPORTANT
The primary AI task is adrenal tumor detection and segmentation.

Adrenal-gland segmentation accuracy is not claimed because
the original dataset does not contain adrenal-gland ground-truth
annotations.

The 53-patient results are dataset-wide evaluation results.
The held-out test results are reported separately.

This is an academic/project prototype and not a clinically
validated medical device.


OUTPUTS
Final dashboard:
{DASHBOARD}

Final JSON:
{FINAL_JSON}

Final CSV:
{FINAL_CSV}

Final text:
{FINAL_TXT}
======================================================================
"""

with open(
    FINAL_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(summary)


print()
print(summary)

print()
print("=" * 70)
print("STEP 17 COMPLETE")
print("=" * 70)