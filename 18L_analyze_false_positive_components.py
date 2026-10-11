from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image
from scipy import ndimage


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"D:\2026\Tumor Detection\Code\001_Adrenal_Tumor_Detection_System"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tumor_focused_resume_training_v2"
    / "best_model.keras"
)

TEST_IMAGE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "unet_dataset"
    / "test"
    / "images"
)

TEST_MASK_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "unet_dataset"
    / "test"
    / "masks"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "metadata"
    / "step18L"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

PREDICTION_THRESHOLD = 0.80

# Candidate minimum component sizes.
CANDIDATE_FILTERS = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200]


# ============================================================
# FUNCTIONS
# ============================================================

def load_image(path):
    image = Image.open(path).convert("L")
    image = image.resize((512, 512))

    array = np.array(image, dtype=np.float32)

    array = array / 255.0

    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)

    return array


def get_components(prediction_mask):
    """
    Return connected components from binary prediction.
    """
    labeled, number = ndimage.label(prediction_mask)

    components = []

    for component_id in range(1, number + 1):
        size = np.count_nonzero(labeled == component_id)

        if size > 0:
            components.append({
                "label": component_id,
                "size": int(size)
            })

    components.sort(
        key=lambda x: x["size"],
        reverse=True
    )

    return components


def apply_component_filter(prediction_mask, minimum_size):
    """
    Remove connected components smaller than minimum_size.
    """
    labeled, number = ndimage.label(prediction_mask)

    filtered = np.zeros_like(prediction_mask, dtype=np.uint8)

    for component_id in range(1, number + 1):

        component_size = np.count_nonzero(
            labeled == component_id
        )

        if component_size >= minimum_size:
            filtered[labeled == component_id] = 1

    return filtered


def calculate_dice(prediction, ground_truth):

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    prediction_sum = prediction.sum()
    ground_truth_sum = ground_truth.sum()

    if prediction_sum + ground_truth_sum == 0:
        return 1.0

    return (
        2.0 * intersection
        / (prediction_sum + ground_truth_sum)
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("STEP 18L")
print("False-Positive Connected Component Analysis")
print("=" * 70)

print()
print("Loading model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print("Model:", model.name)


# ============================================================
# FIND TEST FILES
# ============================================================

image_paths = sorted(TEST_IMAGE_DIR.glob("*.png"))

print()
print("Test images:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No test images found.")


# ============================================================
# RUN INFERENCE
# ============================================================

all_component_sizes = []
false_positive_components = []
true_positive_components = []

slice_results = []

total_gt_positive = 0
total_gt_negative = 0


print()
print("Running inference on held-out test set...")
print()


for index, image_path in enumerate(image_paths, start=1):

    mask_path = TEST_MASK_DIR / image_path.name

    if not mask_path.exists():
        print("Missing mask:", image_path.name)
        continue

    image = load_image(image_path)

    ground_truth = np.array(
        Image.open(mask_path).convert("L")
    )

    ground_truth = ground_truth > 0

    prediction = model.predict(
        image,
        verbose=0
    )[0, :, :, 0]

    prediction_mask = prediction >= PREDICTION_THRESHOLD

    components = get_components(
        prediction_mask
    )

    gt_has_tumor = bool(
        np.count_nonzero(ground_truth) > 0
    )

    if gt_has_tumor:
        total_gt_positive += 1
    else:
        total_gt_negative += 1


    # --------------------------------------------------------
    # Analyze every predicted component
    # --------------------------------------------------------

    for component in components:

        component_size = component["size"]

        all_component_sizes.append(
            component_size
        )

        component_mask = np.zeros(
            prediction_mask.shape,
            dtype=np.uint8
        )

        labeled, _ = ndimage.label(
            prediction_mask
        )

        component_mask[
            labeled == component["label"]
        ] = 1

        overlaps_ground_truth = np.logical_and(
            component_mask.astype(bool),
            ground_truth
        ).any()

        if overlaps_ground_truth:

            true_positive_components.append(
                component_size
            )

        else:

            false_positive_components.append(
                component_size
            )


    # --------------------------------------------------------
    # Largest component before filtering
    # --------------------------------------------------------

    largest_component = 0

    if components:
        largest_component = components[0]["size"]


    slice_results.append({
        "image": image_path.name,
        "ground_truth_tumor": gt_has_tumor,
        "ground_truth_pixels": int(
            np.count_nonzero(ground_truth)
        ),
        "predicted_pixels": int(
            np.count_nonzero(prediction_mask)
        ),
        "components": len(components),
        "largest_component": int(
            largest_component
        ),
        "dice": calculate_dice(
            prediction_mask,
            ground_truth
        )
    })


    if index % 25 == 0 or index == len(image_paths):

        print(
            f"Processed {index}/{len(image_paths)}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("COMPONENT ANALYSIS")
print("=" * 70)

print()
print("Total test slices:", len(slice_results))
print("Ground-truth positive:", total_gt_positive)
print("Ground-truth negative:", total_gt_negative)

print()
print("Total predicted components:", len(all_component_sizes))
print(
    "True-positive components:",
    len(true_positive_components)
)
print(
    "False-positive components:",
    len(false_positive_components)
)


# ============================================================
# FALSE POSITIVE DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("FALSE-POSITIVE COMPONENT SIZES")
print("=" * 70)

if false_positive_components:

    fp_sorted = sorted(
        false_positive_components
    )

    print()
    print("Smallest FP:", fp_sorted[0])
    print("Largest FP:", fp_sorted[-1])
    print(
        "Median FP:",
        int(np.median(fp_sorted))
    )
    print(
        "Mean FP:",
        round(float(np.mean(fp_sorted)), 2)
    )

    print()
    print("First 50 false-positive sizes:")

    print(
        fp_sorted[:50]
    )

else:

    print("No false-positive components found.")


# ============================================================
# TRUE POSITIVE COMPONENT DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("TRUE-POSITIVE COMPONENT SIZES")
print("=" * 70)

if true_positive_components:

    tp_sorted = sorted(
        true_positive_components
    )

    print()
    print("Smallest TP:", tp_sorted[0])
    print("Largest TP:", tp_sorted[-1])
    print(
        "Median TP:",
        int(np.median(tp_sorted))
    )
    print(
        "Mean TP:",
        round(float(np.mean(tp_sorted)), 2)
    )

else:

    print("No true-positive components found.")


# ============================================================
# TEST FILTER THRESHOLDS
# ============================================================

print()
print("=" * 70)
print("MINIMUM COMPONENT SIZE ANALYSIS")
print("=" * 70)

filter_results = []

for minimum_size in CANDIDATE_FILTERS:

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    total_dice = []

    for result in slice_results:

        image_name = result["image"]

        image_path = TEST_IMAGE_DIR / image_name
        mask_path = TEST_MASK_DIR / image_name

        image = load_image(image_path)

        ground_truth = np.array(
            Image.open(mask_path).convert("L")
        )

        ground_truth = ground_truth > 0

        prediction = model.predict(
            image,
            verbose=0
        )[0, :, :, 0]

        raw_prediction = (
            prediction >= PREDICTION_THRESHOLD
        )

        filtered_prediction = apply_component_filter(
            raw_prediction,
            minimum_size
        )

        gt_positive = np.count_nonzero(
            ground_truth
        ) > 0

        predicted_positive = np.count_nonzero(
            filtered_prediction
        ) > 0

        if gt_positive and predicted_positive:
            TP += 1

        elif not gt_positive and not predicted_positive:
            TN += 1

        elif not gt_positive and predicted_positive:
            FP += 1

        elif gt_positive and not predicted_positive:
            FN += 1

        total_dice.append(
            calculate_dice(
                filtered_prediction,
                ground_truth
            )
        )


    accuracy = (
        (TP + TN)
        / max(TP + TN + FP + FN, 1)
    )

    sensitivity = (
        TP
        / max(TP + FN, 1)
    )

    specificity = (
        TN
        / max(TN + FP, 1)
    )

    precision = (
        TP
        / max(TP + FP, 1)
    )

    f1 = (
        2 * precision * sensitivity
        / max(precision + sensitivity, 1e-12)
    )

    mean_dice = float(
        np.mean(total_dice)
    )


    result = {
        "minimum_component_size": minimum_size,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "mean_dice": mean_dice
    }

    filter_results.append(result)


    print(
        f"Minimum {minimum_size:>3} px | "
        f"TP={TP:>3} "
        f"TN={TN:>3} "
        f"FP={FP:>3} "
        f"FN={FN:>3} | "
        f"Sens={sensitivity:.4f} "
        f"Spec={specificity:.4f} "
        f"F1={f1:.4f} "
        f"Dice={mean_dice:.4f}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

report = {
    "prediction_threshold": PREDICTION_THRESHOLD,
    "test_slices": len(slice_results),
    "ground_truth_positive_slices": total_gt_positive,
    "ground_truth_negative_slices": total_gt_negative,
    "false_positive_component_sizes":
        false_positive_components,
    "true_positive_component_sizes":
        true_positive_components,
    "filter_results":
        filter_results
}

report_path = (
    OUTPUT_DIR
    / "false_positive_component_analysis.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print()
print("=" * 70)
print("STEP 18L COMPLETE")
print("=" * 70)

print()
print("Report saved:")
print(report_path)

print()
print(
    "Do NOT change app.py yet."
)

print(
    "Use the results above to select the "
    "minimum component size."
)

