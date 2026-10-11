# ============================================================
# STEP 15
# IMAGE-LEVEL TUMOR DETECTION EVALUATION
# ============================================================
#
# Evaluates the existing U-Net on the unseen test dataset.
#
# NO TRAINING
# NO MODEL MODIFICATION
#
# Input:
#   dataset/unet_dataset/test
#
# Model:
#   Step 16 best_model.keras
#
# Threshold:
#   0.80
#
# Outputs:
#   TP, TN, FP, FN
#   Accuracy
#   Sensitivity / Recall
#   Specificity
#   Precision
#   F1 Score
#   Dice
#   IoU
#
# ============================================================

import os

# ------------------------------------------------------------
# Reduce TensorFlow memory usage
# ------------------------------------------------------------

os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OP"] = "0"

import gc
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# TENSORFLOW THREAD SETTINGS
# ============================================================

try:

    tf.config.threading.set_intra_op_parallelism_threads(1)

    tf.config.threading.set_inter_op_parallelism_threads(1)

except Exception:

    pass


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tumor_focused_resume_training_v2"
    / "best_model.keras"
)

TEST_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "unet_dataset"
    / "test"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "step15"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 512

THRESHOLD = 0.80

MIN_COMPONENT_SIZE = 50


# ============================================================
# START
# ============================================================

print()
print("=" * 72)
print("STEP 15 - IMAGE-LEVEL TUMOR DETECTION EVALUATION")
print("=" * 72)
print()


# ============================================================
# CHECK PATHS
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"Test dataset not found:\n{TEST_DIR}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading final U-Net model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print()
print("MODEL LOAD SUCCESS")
print(
    "Model input:",
    model.input_shape
)
print(
    "Model output:",
    model.output_shape
)
print()


# ============================================================
# FIND IMAGE/MASK PAIRS
# ============================================================

image_files = sorted(
    TEST_DIR.glob("images/*.png")
)

mask_files = sorted(
    TEST_DIR.glob("masks/*.png")
)

print(
    "Test images:",
    len(image_files)
)

print(
    "Test masks:",
    len(mask_files)
)

print()


if len(image_files) == 0:

    raise RuntimeError(
        "No test images found."
    )


if len(image_files) != len(mask_files):

    raise RuntimeError(
        "Number of test images and masks do not match."
    )


# ============================================================
# BUILD MASK LOOKUP
# ============================================================

mask_lookup = {

    path.name: path

    for path in mask_files

}


# ============================================================
# CONNECTED COMPONENT FILTER
# ============================================================

def largest_component(
    mask
):

    # --------------------------------------------------------
    # Lightweight 2D flood-fill.
    # Avoids creating large scipy label arrays.
    # --------------------------------------------------------

    height, width = mask.shape

    visited = np.zeros(
        mask.shape,
        dtype=np.uint8
    )

    largest = []

    component_count = 0

    for y in range(height):

        for x in range(width):

            if mask[y, x] == 0:

                continue

            if visited[y, x] != 0:

                continue

            component_count += 1

            stack = [(y, x)]

            visited[y, x] = 1

            component = []

            while stack:

                cy, cx = stack.pop()

                component.append(
                    (cy, cx)
                )

                for dy in (
                    -1,
                    0,
                    1
                ):

                    for dx in (
                        -1,
                        0,
                        1
                    ):

                        if (
                            dy == 0
                            and
                            dx == 0
                        ):

                            continue

                        ny = cy + dy
                        nx = cx + dx

                        if (
                            ny < 0
                            or ny >= height
                            or
                            nx < 0
                            or nx >= width
                        ):

                            continue

                        if mask[
                            ny,
                            nx
                        ] == 0:

                            continue

                        if visited[
                            ny,
                            nx
                        ] != 0:

                            continue

                        visited[
                            ny,
                            nx
                        ] = 1

                        stack.append(
                            (ny, nx)
                        )

            if (
                len(component)
                >= MIN_COMPONENT_SIZE
            ):

                if len(component) > len(
                    largest
                ):

                    largest = component

    result = np.zeros(
        mask.shape,
        dtype=np.uint8
    )

    for y, x in largest:

        result[y, x] = 1

    del visited

    return (
        result,
        component_count,
        len(largest)
    )


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_segmentation_metrics(
    prediction,
    ground_truth
):

    prediction = (
        prediction > 0
    )

    ground_truth = (
        ground_truth > 0
    )

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    prediction_sum = prediction.sum()

    ground_truth_sum = ground_truth.sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    if (
        prediction_sum
        +
        ground_truth_sum
        ==
        0
    ):

        dice = 1.0

    else:

        dice = (
            2.0
            * intersection
            /
            (
                prediction_sum
                +
                ground_truth_sum
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

    return (
        float(dice),
        float(iou)
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

true_positive = 0

true_negative = 0

false_positive = 0

false_negative = 0


# ============================================================
# SEGMENTATION METRICS
# ============================================================

dice_values = []

iou_values = []

positive_dice_values = []

positive_iou_values = []


# ============================================================
# IMAGE COUNTS
# ============================================================

ground_truth_positive = 0

ground_truth_negative = 0

prediction_positive = 0

prediction_negative = 0


# ============================================================
# PROCESS TEST SET
# ============================================================

total = len(
    image_files
)


for index, image_path in enumerate(
    image_files,
    start=1
):

    # --------------------------------------------------------
    # Find matching mask
    # --------------------------------------------------------

    mask_path = mask_lookup.get(
        image_path.name
    )

    if mask_path is None:

        print(
            f"WARNING: mask not found for "
            f"{image_path.name}"
        )

        continue

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        color_mode="grayscale",
        target_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image_array = (
        tf.keras.utils.img_to_array(
            image
        )
    )

    image_array = (
        image_array
        / 255.0
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    model_input = (
        image_array[
            np.newaxis,
            ...
        ]
    )

    # --------------------------------------------------------
    # U-Net inference
    # --------------------------------------------------------

    output = model(
        model_input,
        training=False
    )

    probability = (
        output[
            0,
            :,
            :,
            0
        ].numpy()
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    raw_prediction = (
        probability
        >= THRESHOLD
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Connected component filtering
    # --------------------------------------------------------

    prediction_mask, _, _ = (
        largest_component(
            raw_prediction
        )
    )

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    gt_image = tf.keras.utils.load_img(
        mask_path,
        color_mode="grayscale",
        target_size=(
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    ground_truth = (
        tf.keras.utils.img_to_array(
            gt_image
        )[:, :, 0]
    )

    ground_truth = (
        ground_truth > 127
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Image-level classification
    #
    # Positive = at least one tumor pixel
    # --------------------------------------------------------

    gt_positive = (
        np.any(
            ground_truth > 0
        )
    )

    pred_positive = (
        np.any(
            prediction_mask > 0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    if gt_positive and pred_positive:

        true_positive += 1

    elif (
        not gt_positive
        and
        not pred_positive
    ):

        true_negative += 1

    elif (
        not gt_positive
        and
        pred_positive
    ):

        false_positive += 1

    elif (
        gt_positive
        and
        not pred_positive
    ):

        false_negative += 1

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    if gt_positive:

        ground_truth_positive += 1

    else:

        ground_truth_negative += 1

    if pred_positive:

        prediction_positive += 1

    else:

        prediction_negative += 1

    # --------------------------------------------------------
    # Segmentation metrics
    # --------------------------------------------------------

    dice, iou = (
        calculate_segmentation_metrics(
            prediction_mask,
            ground_truth
        )
    )

    dice_values.append(
        dice
    )

    iou_values.append(
        iou
    )

    if gt_positive:

        positive_dice_values.append(
            dice
        )

        positive_iou_values.append(
            iou
        )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index == 1
        or
        index % 25 == 0
        or
        index == total
    ):

        print(
            f"Processed "
            f"{index}/{total}"
        )

    # --------------------------------------------------------
    # Release memory
    # --------------------------------------------------------

    del image
    del image_array
    del model_input
    del output
    del probability
    del raw_prediction
    del prediction_mask
    del gt_image
    del ground_truth

    gc.collect()


# ============================================================
# CALCULATE CLASSIFICATION METRICS
# ============================================================

total_samples = (
    true_positive
    +
    true_negative
    +
    false_positive
    +
    false_negative
)


if (
    true_positive
    +
    false_positive
) > 0:

    precision = (
        true_positive
        /
        (
            true_positive
            +
            false_positive
        )
    )

else:

    precision = 0.0


if (
    true_positive
    +
    false_negative
) > 0:

    sensitivity = (
        true_positive
        /
        (
            true_positive
            +
            false_negative
        )
    )

else:

    sensitivity = 0.0


if (
    true_negative
    +
    false_positive
) > 0:

    specificity = (
        true_negative
        /
        (
            true_negative
            +
            false_positive
        )
    )

else:

    specificity = 0.0


if total_samples > 0:

    accuracy = (
        (
            true_positive
            +
            true_negative
        )
        /
        total_samples
    )

else:

    accuracy = 0.0


if (
    precision
    +
    sensitivity
) > 0:

    f1 = (
        2.0
        * precision
        * sensitivity
        /
        (
            precision
            +
            sensitivity
        )
    )

else:

    f1 = 0.0


# ============================================================
# SEGMENTATION SUMMARY
# ============================================================

mean_dice = (
    float(
        np.mean(
            dice_values
        )
    )
    if dice_values
    else 0.0
)


mean_iou = (
    float(
        np.mean(
            iou_values
        )
    )
    if iou_values
    else 0.0
)


positive_mean_dice = (
    float(
        np.mean(
            positive_dice_values
        )
    )
    if positive_dice_values
    else 0.0
)


positive_mean_iou = (
    float(
        np.mean(
            positive_iou_values
        )
    )
    if positive_iou_values
    else 0.0
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 72)
print("STEP 15 - FINAL EVALUATION")
print("=" * 72)

print()

print(
    f"Test samples             : {total_samples}"
)

print(
    f"Ground-truth positive    : "
    f"{ground_truth_positive}"
)

print(
    f"Ground-truth negative    : "
    f"{ground_truth_negative}"
)

print(
    f"Predicted positive       : "
    f"{prediction_positive}"
)

print(
    f"Predicted negative       : "
    f"{prediction_negative}"
)

print()

print(
    f"True Positive            : "
    f"{true_positive}"
)

print(
    f"True Negative            : "
    f"{true_negative}"
)

print(
    f"False Positive           : "
    f"{false_positive}"
)

print(
    f"False Negative           : "
    f"{false_negative}"
)

print()

print(
    f"Accuracy                 : "
    f"{accuracy:.6f}"
)

print(
    f"Sensitivity / Recall     : "
    f"{sensitivity:.6f}"
)

print(
    f"Specificity              : "
    f"{specificity:.6f}"
)

print(
    f"Precision                : "
    f"{precision:.6f}"
)

print(
    f"F1 Score                 : "
    f"{f1:.6f}"
)

print()

print(
    f"Mean Dice - all slices   : "
    f"{mean_dice:.6f}"
)

print(
    f"Mean IoU - all slices    : "
    f"{mean_iou:.6f}"
)

print(
    f"Mean Dice - tumor slices : "
    f"{positive_mean_dice:.6f}"
)

print(
    f"Mean IoU - tumor slices  : "
    f"{positive_mean_iou:.6f}"
)

print()

print(
    f"Threshold                : "
    f"{THRESHOLD:.2f}"
)

print()


# ============================================================
# SAVE REPORT
# ============================================================

report = {

    "step":
        "15",

    "purpose":
        "Image-level tumor detection evaluation",

    "model":
        str(MODEL_PATH),

    "test_dataset":
        str(TEST_DIR),

    "test_samples":
        total_samples,

    "threshold":
        THRESHOLD,

    "ground_truth_positive":
        ground_truth_positive,

    "ground_truth_negative":
        ground_truth_negative,

    "predicted_positive":
        prediction_positive,

    "predicted_negative":
        prediction_negative,

    "true_positive":
        true_positive,

    "true_negative":
        true_negative,

    "false_positive":
        false_positive,

    "false_negative":
        false_negative,

    "accuracy":
        accuracy,

    "sensitivity":
        sensitivity,

    "specificity":
        specificity,

    "precision":
        precision,

    "f1_score":
        f1,

    "mean_dice_all_slices":
        mean_dice,

    "mean_iou_all_slices":
        mean_iou,

    "mean_dice_tumor_slices":
        positive_mean_dice,

    "mean_iou_tumor_slices":
        positive_mean_iou,

    "status":
        "PASS"
}


report_path = (
    OUTPUT_DIR
    / "image_level_evaluation_report.json"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4
    )


# ============================================================
# TEXT REPORT
# ============================================================

text_path = (
    OUTPUT_DIR
    / "image_level_evaluation_summary.txt"
)


with open(
    text_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 15 - IMAGE-LEVEL TUMOR DETECTION EVALUATION\n"
    )

    f.write(
        "=" * 65
        + "\n\n"
    )

    f.write(
        f"Test samples: {total_samples}\n"
    )

    f.write(
        f"Ground-truth positive: "
        f"{ground_truth_positive}\n"
    )

    f.write(
        f"Ground-truth negative: "
        f"{ground_truth_negative}\n"
    )

    f.write(
        f"Predicted positive: "
        f"{prediction_positive}\n"
    )

    f.write(
        f"Predicted negative: "
        f"{prediction_negative}\n\n"
    )

    f.write(
        f"True Positive: {true_positive}\n"
    )

    f.write(
        f"True Negative: {true_negative}\n"
    )

    f.write(
        f"False Positive: {false_positive}\n"
    )

    f.write(
        f"False Negative: {false_negative}\n\n"
    )

    f.write(
        f"Accuracy: {accuracy:.6f}\n"
    )

    f.write(
        f"Sensitivity: {sensitivity:.6f}\n"
    )

    f.write(
        f"Specificity: {specificity:.6f}\n"
    )

    f.write(
        f"Precision: {precision:.6f}\n"
    )

    f.write(
        f"F1 Score: {f1:.6f}\n\n"
    )

    f.write(
        f"Mean Dice - all slices: "
        f"{mean_dice:.6f}\n"
    )

    f.write(
        f"Mean IoU - all slices: "
        f"{mean_iou:.6f}\n"
    )

    f.write(
        f"Mean Dice - tumor slices: "
        f"{positive_mean_dice:.6f}\n"
    )

    f.write(
        f"Mean IoU - tumor slices: "
        f"{positive_mean_iou:.6f}\n\n"
    )

    f.write(
        f"Threshold: {THRESHOLD:.2f}\n\n"
    )

    f.write(
        "STEP 33 STATUS: PASS\n"
    )


print(
    "JSON report:"
)

print(
    report_path
)

print()

print(
    "Text summary:"
)

print(
    text_path
)

print()

print(
    "STEP 15 STATUS: PASS"
)

print()


# ============================================================
# CLEANUP
# ============================================================

del model

gc.collect()