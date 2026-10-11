import os
import json
import csv
import time
import numpy as np
import nibabel as nib
import tensorflow as tf

from scipy import ndimage
from pathlib import Path


# ============================================================
# STEP 16
# FAST FINAL 53-PATIENT EVALUATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CT_DIR = PROJECT_ROOT / "dataset" / "ct_volumes"
GT_DIR = PROJECT_ROOT / "dataset" / "tumor_ground_truth"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tumor_focused_resume_training_v2"
    / "best_model.keras"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "step16"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "final_53_patient_evaluation.csv"
JSON_PATH = OUTPUT_DIR / "final_53_patient_evaluation.json"
SUMMARY_PATH = OUTPUT_DIR / "FINAL_53_PATIENT_EVALUATION_SUMMARY.txt"

THRESHOLD = 0.80

# Batch size for CPU inference.
# 8 is safer for RAM than a very large batch.
BATCH_SIZE = 8


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_ct(volume):
    """
    Same CT preprocessing used during model development.

    Window:
        Center = 40 HU
        Width  = 400 HU

    Range:
        -160 HU to +240 HU

    Normalize to [0, 1].
    """

    lower = -160.0
    upper = 240.0

    volume = np.clip(
        volume,
        lower,
        upper
    )

    volume = (
        volume - lower
    ) / (
        upper - lower
    )

    return volume.astype(np.float32)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

def largest_component_2d(mask):
    """
    Keep largest connected component in a 2D slice.
    """

    mask = mask.astype(bool)

    if not np.any(mask):
        return np.zeros_like(
            mask,
            dtype=bool
        )

    labeled, num = ndimage.label(mask)

    if num == 0:
        return np.zeros_like(
            mask,
            dtype=bool
        )

    sizes = np.bincount(
        labeled.ravel()
    )

    largest_label = (
        np.argmax(sizes[1:]) + 1
    )

    return labeled == largest_label


def largest_component_3d(mask):
    """
    Keep largest connected component in 3D.
    """

    mask = mask.astype(bool)

    if not np.any(mask):
        return np.zeros_like(
            mask,
            dtype=bool
        )

    structure = ndimage.generate_binary_structure(
        3,
        1
    )

    labeled, num = ndimage.label(
        mask,
        structure=structure
    )

    if num == 0:
        return np.zeros_like(
            mask,
            dtype=bool
        )

    sizes = np.bincount(
        labeled.ravel()
    )

    largest_label = (
        np.argmax(sizes[1:]) + 1
    )

    return labeled == largest_label


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(gt, pred):

    gt = gt.astype(bool)
    pred = pred.astype(bool)

    tp = np.logical_and(
        gt,
        pred
    ).sum()

    tn = np.logical_and(
        ~gt,
        ~pred
    ).sum()

    fp = np.logical_and(
        ~gt,
        pred
    ).sum()

    fn = np.logical_and(
        gt,
        ~pred
    ).sum()

    intersection = tp

    union = np.logical_or(
        gt,
        pred
    ).sum()

    if gt.sum() + pred.sum() == 0:
        dice = 1.0
    else:
        dice = (
            2.0 * intersection
            /
            (
                gt.sum()
                +
                pred.sum()
            )
        )

    if union == 0:
        iou = 1.0
    else:
        iou = (
            intersection
            /
            union
        )

    sensitivity = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    specificity = (
        tn / (tn + fp)
        if tn + fp > 0
        else 0.0
    )

    accuracy = (
        (tp + tn)
        /
        (tp + tn + fp + fn)
        if tp + tn + fp + fn > 0
        else 0.0
    )

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),

        "dice": float(dice),
        "iou": float(iou),
        "sensitivity": float(sensitivity),
        "precision": float(precision),
        "specificity": float(specificity),
        "accuracy": float(accuracy)
    }


# ============================================================
# MEASUREMENTS
# ============================================================

def tumor_volume_cm3(
    mask,
    spacing
):

    voxel_volume_mm3 = (
        float(spacing[0])
        *
        float(spacing[1])
        *
        float(spacing[2])
    )

    voxels = int(
        mask.sum()
    )

    return float(
        voxels
        *
        voxel_volume_mm3
        /
        1000.0
    )


def bounding_box_dimension_mm(
    mask,
    spacing
):

    coords = np.argwhere(mask)

    if len(coords) == 0:
        return 0.0

    min_xyz = coords.min(
        axis=0
    )

    max_xyz = coords.max(
        axis=0
    )

    size_voxels = (
        max_xyz
        -
        min_xyz
        +
        1
    )

    size_mm = (
        size_voxels.astype(float)
        *
        np.asarray(
            spacing,
            dtype=float
        )
    )

    return float(
        np.max(size_mm)
    )


# ============================================================
# PATIENT EVALUATION
# ============================================================

def evaluate_patient(
    model,
    patient_id
):

    ct_path = (
        CT_DIR
        /
        patient_id
        /
        f"{patient_id}_original_ct.nii.gz"
    )

    gt_path = (
        GT_DIR
        /
        patient_id
        /
        f"{patient_id}_tumor_mask.nii.gz"
    )

    if not ct_path.exists():
        raise FileNotFoundError(
            f"CT not found: {ct_path}"
        )

    if not gt_path.exists():
        raise FileNotFoundError(
            f"GT not found: {gt_path}"
        )

    # --------------------------------------------------------
    # Load CT
    # --------------------------------------------------------

    ct_nii = nib.load(
        str(ct_path)
    )

    gt_nii = nib.load(
        str(gt_path)
    )

    ct = ct_nii.get_fdata(
        dtype=np.float32
    )

    gt = (
        gt_nii.get_fdata()
        > 0
    )

    spacing = (
        ct_nii.header
        .get_zooms()[:3]
    )

    # --------------------------------------------------------
    # Geometry validation
    # --------------------------------------------------------

    if ct.shape != gt.shape:

        raise ValueError(
            f"{patient_id}: "
            f"CT/GT shape mismatch "
            f"{ct.shape} vs {gt.shape}"
        )

    gt_spacing = (
        gt_nii.header
        .get_zooms()[:3]
    )

    if not np.allclose(
        spacing,
        gt_spacing,
        atol=1e-4
    ):

        raise ValueError(
            f"{patient_id}: "
            "CT/GT spacing mismatch"
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    processed = preprocess_ct(
        ct
    )

    # --------------------------------------------------------
    # Convert volume to slices
    #
    # CT shape:
    # (H, W, Z)
    #
    # Model input:
    # (Z, H, W, 1)
    # --------------------------------------------------------

    slices = np.transpose(
        processed,
        (2, 0, 1)
    )

    model_input = (
        slices[..., np.newaxis]
    )

    # --------------------------------------------------------
    # BATCHED MODEL INFERENCE
    # --------------------------------------------------------

    probabilities = []

    total_slices = model_input.shape[0]

    for start in range(
        0,
        total_slices,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            total_slices
        )

        batch = model_input[
            start:end
        ]

        batch_probability = (
            model.predict(
                batch,
                batch_size=BATCH_SIZE,
                verbose=0
            )
        )

        probabilities.append(
            batch_probability[
                ..., 0
            ]
        )

    probabilities = np.concatenate(
        probabilities,
        axis=0
    )

    # Convert back:
    # (Z,H,W) -> (H,W,Z)

    probabilities = np.transpose(
        probabilities,
        (1, 2, 0)
    )

    # --------------------------------------------------------
    # THRESHOLD + 2D COMPONENT
    # --------------------------------------------------------

    prediction = np.zeros_like(
        gt,
        dtype=bool
    )

    for z in range(
        prediction.shape[2]
    ):

        binary = (
            probabilities[:, :, z]
            >= THRESHOLD
        )

        prediction[:, :, z] = (
            largest_component_2d(
                binary
            )
        )

    # --------------------------------------------------------
    # 3D COMPONENT
    # --------------------------------------------------------

    prediction = (
        largest_component_3d(
            prediction
        )
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        gt,
        prediction
    )

    gt_voxels = int(
        gt.sum()
    )

    pred_voxels = int(
        prediction.sum()
    )

    gt_volume = tumor_volume_cm3(
        gt,
        spacing
    )

    pred_volume = tumor_volume_cm3(
        prediction,
        spacing
    )

    if gt_volume > 0:

        volume_error = (
            abs(
                pred_volume
                -
                gt_volume
            )
            /
            gt_volume
            *
            100.0
        )

    else:

        volume_error = 0.0

    gt_bbox = (
        bounding_box_dimension_mm(
            gt,
            spacing
        )
    )

    pred_bbox = (
        bounding_box_dimension_mm(
            prediction,
            spacing
        )
    )

    # --------------------------------------------------------
    # TUMOR DETECTION
    # --------------------------------------------------------

    gt_positive = (
        gt_voxels > 0
    )

    predicted_positive = (
        pred_voxels > 0
    )

    detection_correct = (
        gt_positive
        ==
        predicted_positive
    )

    # --------------------------------------------------------
    # POSITIVE SLICES
    # --------------------------------------------------------

    gt_slices = np.where(
        np.any(
            gt,
            axis=(0, 1)
        )
    )[0]

    pred_slices = np.where(
        np.any(
            prediction,
            axis=(0, 1)
        )
    )[0]

    if len(gt_slices) > 0:

        gt_slice_start = int(
            gt_slices[0]
        )

        gt_slice_end = int(
            gt_slices[-1]
        )

    else:

        gt_slice_start = None
        gt_slice_end = None

    if len(pred_slices) > 0:

        pred_slice_start = int(
            pred_slices[0]
        )

        pred_slice_end = int(
            pred_slices[-1]
        )

    else:

        pred_slice_start = None
        pred_slice_end = None

    return {
        "patient_id": patient_id,

        "status": "PASS",

        "ct_shape": list(
            ct.shape
        ),

        "spacing_mm": [
            float(spacing[0]),
            float(spacing[1]),
            float(spacing[2])
        ],

        "gt_voxels": gt_voxels,

        "predicted_voxels":
            pred_voxels,

        "gt_tumor_present":
            bool(gt_positive),

        "predicted_tumor_present":
            bool(predicted_positive),

        "detection_correct":
            bool(detection_correct),

        "gt_volume_cm3":
            gt_volume,

        "predicted_volume_cm3":
            pred_volume,

        "volume_error_percent":
            float(volume_error),

        "gt_maximum_bbox_dimension_mm":
            gt_bbox,

        "predicted_maximum_bbox_dimension_mm":
            pred_bbox,

        "gt_positive_slice_start":
            gt_slice_start,

        "gt_positive_slice_end":
            gt_slice_end,

        "predicted_positive_slice_start":
            pred_slice_start,

        "predicted_positive_slice_end":
            pred_slice_end,

        **metrics
    }


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print("=" * 70)
    print("STEP 16 - FAST FINAL 53-PATIENT EVALUATION")
    print("=" * 70)

    print()
    print("Batch size:", BATCH_SIZE)
    print("Threshold:", THRESHOLD)

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not CT_DIR.exists():

        raise FileNotFoundError(
            f"CT directory not found:\n{CT_DIR}"
        )

    if not GT_DIR.exists():

        raise FileNotFoundError(
            f"GT directory not found:\n{GT_DIR}"
        )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()
    print("Loading final U-Net model...")

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    print("Model loaded.")

    print(
        "Input:",
        model.input_shape
    )

    print(
        "Output:",
        model.output_shape
    )

    print(
        "Parameters:",
        f"{model.count_params():,}"
    )

    # --------------------------------------------------------
    # PATIENTS
    # --------------------------------------------------------

    patient_dirs = sorted([
        p.name
        for p in CT_DIR.iterdir()
        if p.is_dir()
        and p.name.startswith(
            "Adrenal_Ki67_Seg_"
        )
    ])

    print()
    print(
        "Patients found:",
        len(patient_dirs)
    )

    if len(patient_dirs) != 53:

        print(
            "WARNING: Expected 53 patients."
        )

   
    existing_results = {}

    if CSV_PATH.exists():

        print()
        print(
            "Existing evaluation CSV found."
        )

        try:

            with open(
                CSV_PATH,
                "r",
                encoding="utf-8"
            ) as f:

                reader = csv.DictReader(f)

                for row in reader:

                    if row.get("status") == "PASS":

                        existing_results[
                            row["patient_id"]
                        ] = row

            print(
                "Previously completed patients:",
                len(existing_results)
            )

        except Exception as e:

            print(
                "Could not read old CSV.",
                "Starting fresh."
            )

            existing_results = {}

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    results = []

    for index, patient_id in enumerate(
        patient_dirs,
        start=1
    ):

        print()
        print(
            f"[{index:02d}/{len(patient_dirs):02d}] "
            f"{patient_id}"
        )

        

        patient_start = time.time()

        try:

            result = evaluate_patient(
                model,
                patient_id
            )

            elapsed = (
                time.time()
                -
                patient_start
            )

            result[
                "processing_time_sec"
            ] = float(elapsed)

            results.append(
                result
            )

            print(
                f"  Dice        : "
                f"{result['dice']:.4f}"
            )

            print(
                f"  IoU         : "
                f"{result['iou']:.4f}"
            )

            print(
                f"  Sensitivity : "
                f"{result['sensitivity']:.4f}"
            )

            print(
                f"  Precision   : "
                f"{result['precision']:.4f}"
            )

            print(
                f"  GT volume   : "
                f"{result['gt_volume_cm3']:.2f} cm3"
            )

            print(
                f"  Pred volume : "
                f"{result['predicted_volume_cm3']:.2f} cm3"
            )

            print(
                f"  Time        : "
                f"{elapsed:.2f} sec"
            )

        except Exception as e:

            print(
                "  ERROR:",
                str(e)
            )

            results.append({
                "patient_id":
                    patient_id,
                "status":
                    "FAIL",
                "error":
                    str(e)
            })

    # --------------------------------------------------------
    # SUCCESSFUL
    # --------------------------------------------------------

    successful = [
        r
        for r in results
        if r.get("status") == "PASS"
    ]

    failed = [
        r
        for r in results
        if r.get("status") == "FAIL"
    ]

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(
        "Successful:",
        len(successful)
    )

    print(
        "Failed:",
        len(failed)
    )

    if not successful:

        raise RuntimeError(
            "No successful evaluations."
        )

    # --------------------------------------------------------
    # MEANS
    # --------------------------------------------------------

    def mean_metric(name):

        return float(
            np.mean([
                r[name]
                for r in successful
            ])
        )

    mean_dice = mean_metric(
        "dice"
    )

    mean_iou = mean_metric(
        "iou"
    )

    mean_sensitivity = mean_metric(
        "sensitivity"
    )

    mean_precision = mean_metric(
        "precision"
    )

    mean_specificity = mean_metric(
        "specificity"
    )

    mean_accuracy = mean_metric(
        "accuracy"
    )

    mean_gt_volume = mean_metric(
        "gt_volume_cm3"
    )

    mean_pred_volume = mean_metric(
        "predicted_volume_cm3"
    )

    mean_volume_error = mean_metric(
        "volume_error_percent"
    )

    mean_gt_bbox = mean_metric(
        "gt_maximum_bbox_dimension_mm"
    )

    mean_pred_bbox = mean_metric(
        "predicted_maximum_bbox_dimension_mm"
    )

    # --------------------------------------------------------
    # DETECTION
    # --------------------------------------------------------

    gt_positive = sum(
        r["gt_tumor_present"]
        for r in successful
    )

    predicted_positive = sum(
        r["predicted_tumor_present"]
        for r in successful
    )

    detection_correct = sum(
        r["detection_correct"]
        for r in successful
    )

    detection_accuracy = (
        detection_correct
        /
        len(successful)
    )

    # --------------------------------------------------------
    # BEST / WORST
    # --------------------------------------------------------

    best_patient = max(
        successful,
        key=lambda x: x["dice"]
    )

    worst_patient = min(
        successful,
        key=lambda x: x["dice"]
    )

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    fieldnames = [
        "patient_id",
        "status",
        "ct_shape",
        "spacing_mm",

        "gt_voxels",
        "predicted_voxels",

        "gt_tumor_present",
        "predicted_tumor_present",
        "detection_correct",

        "gt_volume_cm3",
        "predicted_volume_cm3",
        "volume_error_percent",

        "gt_maximum_bbox_dimension_mm",
        "predicted_maximum_bbox_dimension_mm",

        "gt_positive_slice_start",
        "gt_positive_slice_end",

        "predicted_positive_slice_start",
        "predicted_positive_slice_end",

        "TP",
        "TN",
        "FP",
        "FN",

        "dice",
        "iou",
        "sensitivity",
        "precision",
        "specificity",
        "accuracy",

        "processing_time_sec"
    ]

    with open(
        CSV_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in successful:

            writer.writerow({
                key:
                    result.get(
                        key,
                        ""
                    )
                for key in fieldnames
            })

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    total_time = (
        time.time()
        -
        overall_start
    )

    report = {

        "project":
            "AI-Based Adrenal Tumor "
            "Detection and Segmentation System",

        "step":
            "16",

        "evaluation_type":
            "Final 53-patient dataset-wide evaluation",

        "model":
            str(MODEL_PATH),

        "model_parameters":
            int(model.count_params()),

        "prediction_threshold":
            THRESHOLD,

        "batch_size":
            BATCH_SIZE,

        "post_processing": [
            "Largest connected component per 2D slice",
            "Largest connected component in 3D volume"
        ],

        "patients_found":
            len(patient_dirs),

        "patients_successful":
            len(successful),

        "patients_failed":
            len(failed),

        "gt_positive_patients":
            int(gt_positive),

        "predicted_positive_patients":
            int(predicted_positive),

        "detection_accuracy":
            float(
                detection_accuracy
            ),

        "mean_metrics": {

            "dice":
                mean_dice,

            "iou":
                mean_iou,

            "sensitivity":
                mean_sensitivity,

            "precision":
                mean_precision,

            "specificity":
                mean_specificity,

            "accuracy":
                mean_accuracy,

            "gt_tumor_volume_cm3":
                mean_gt_volume,

            "predicted_tumor_volume_cm3":
                mean_pred_volume,

            "volume_error_percent":
                mean_volume_error,

            "gt_maximum_bbox_dimension_mm":
                mean_gt_bbox,

            "predicted_maximum_bbox_dimension_mm":
                mean_pred_bbox
        },

        "best_patient": {
            "patient_id":
                best_patient["patient_id"],
            "dice":
                best_patient["dice"],
            "iou":
                best_patient["iou"]
        },

        "worst_patient": {
            "patient_id":
                worst_patient["patient_id"],
            "dice":
                worst_patient["dice"],
            "iou":
                worst_patient["iou"]
        },

        "failed_patients":
            failed,

        "results":
            results,

        "total_processing_time_sec":
            float(total_time),

        "status":
            "PASS"
            if len(successful)
            ==
            len(patient_dirs)
            else
            "PARTIAL"
    }

    with open(
        JSON_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # TEXT SUMMARY
    # --------------------------------------------------------

    summary = f"""
============================================================
STEP 16 - FINAL 53-PATIENT EVALUATION
============================================================

Project:
AI-Based Adrenal Tumor Detection and Segmentation System

Model:
{MODEL_PATH}

Model parameters:
{model.count_params():,}

Prediction threshold:
{THRESHOLD}

Batch size:
{BATCH_SIZE}

Post-processing:
- Largest connected component per 2D slice
- Largest connected component in 3D volume

------------------------------------------------------------
DATASET
------------------------------------------------------------

Patients found:
{len(patient_dirs)}

Successful:
{len(successful)}

Failed:
{len(failed)}

Ground-truth positive patients:
{gt_positive}

Predicted positive patients:
{predicted_positive}

Tumor presence detection accuracy:
{detection_accuracy * 100:.2f}%

------------------------------------------------------------
SEGMENTATION
------------------------------------------------------------

Mean Dice:
{mean_dice:.6f}

Mean IoU:
{mean_iou:.6f}

Mean Sensitivity:
{mean_sensitivity:.6f}

Mean Precision:
{mean_precision:.6f}

Mean Specificity:
{mean_specificity:.6f}

Mean Accuracy:
{mean_accuracy:.6f}

------------------------------------------------------------
TUMOR MEASUREMENTS
------------------------------------------------------------

Mean GT tumor volume:
{mean_gt_volume:.4f} cm3

Mean predicted tumor volume:
{mean_pred_volume:.4f} cm3

Mean absolute volume error:
{mean_volume_error:.4f}%

Mean GT maximum bounding-box dimension:
{mean_gt_bbox:.4f} mm

Mean predicted maximum bounding-box dimension:
{mean_pred_bbox:.4f} mm

------------------------------------------------------------
PATIENT EXTREMES
------------------------------------------------------------

Best Dice:
{best_patient["patient_id"]}
Dice = {best_patient["dice"]:.6f}

Worst Dice:
{worst_patient["patient_id"]}
Dice = {worst_patient["dice"]:.6f}

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

CSV:
{CSV_PATH}

JSON:
{JSON_PATH}

SUMMARY:
{SUMMARY_PATH}

------------------------------------------------------------
NOTE
------------------------------------------------------------

The tumor ground truth originates from the original TCIA
DICOM SEG annotations.

The model performs adrenal tumor detection and segmentation.

The dataset does not contain ground-truth adrenal-gland
segmentation, therefore this evaluation does not claim
adrenal-gland segmentation accuracy.

This is an academic/project evaluation and not a clinical
validation.

Total processing time:
{total_time:.2f} seconds

============================================================
"""

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(summary)

    print(summary)

    print()
    print("STEP 16 FINISHED.")


if __name__ == "__main__":
    main()