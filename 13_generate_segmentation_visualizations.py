from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# ============================================================
# STEP 19 - SEGMENTATION VISUALIZATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "unet_dataset"
)

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
    / "step13"
    / "visualizations"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 512

# Number of visual examples to generate
NUMBER_OF_EXAMPLES = 10

# Prediction threshold
PREDICTION_THRESHOLD = 0.5


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 13- SEGMENTATION VISUALIZATION")
print("=" * 70)

print()
print(f"TensorFlow version : {tf.__version__}")
print(f"Model              : {MODEL_PATH}")
print(f"Test dataset       : {DATASET_ROOT / 'test'}")


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise SystemExit(
        f"ERROR: Best model not found:\n{MODEL_PATH}"
    )

TEST_IMAGE_DIR = DATASET_ROOT / "test" / "images"
TEST_MASK_DIR = DATASET_ROOT / "test" / "masks"

if not TEST_IMAGE_DIR.exists():
    raise SystemExit(
        f"ERROR: Test image directory not found:\n{TEST_IMAGE_DIR}"
    )

if not TEST_MASK_DIR.exists():
    raise SystemExit(
        f"ERROR: Test mask directory not found:\n{TEST_MASK_DIR}"
    )

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND TEST PAIRS
# ============================================================

image_files = sorted(
    TEST_IMAGE_DIR.glob("*.png")
)

pairs = []

for image_path in image_files:

    mask_path = TEST_MASK_DIR / image_path.name

    if mask_path.exists():
        pairs.append(
            (image_path, mask_path)
        )


if not pairs:
    raise SystemExit(
        "ERROR: No valid test image/mask pairs found."
    )

print()
print(f"Test image/mask pairs : {len(pairs)}")


# ============================================================
# SELECT REPRESENTATIVE SLICES
# ============================================================

# Prefer slices containing tumor.
tumor_pairs = []
background_pairs = []

for image_path, mask_path in pairs:

    mask = plt.imread(
        mask_path
    )

    if mask.ndim == 3:
        mask = mask[:, :, 0]

    if np.any(mask > 0.5):
        tumor_pairs.append(
            (image_path, mask_path)
        )
    else:
        background_pairs.append(
            (image_path, mask_path)
        )


print(
    f"Tumor-containing pairs : {len(tumor_pairs)}"
)

print(
    f"Background pairs       : {len(background_pairs)}"
)


# Select tumor examples first.
selected_pairs = []

if tumor_pairs:

    step = max(
        1,
        len(tumor_pairs) // NUMBER_OF_EXAMPLES
    )

    selected_pairs = tumor_pairs[::step][
        :NUMBER_OF_EXAMPLES
    ]

# If fewer than requested, add background examples.
if len(selected_pairs) < NUMBER_OF_EXAMPLES:

    remaining = NUMBER_OF_EXAMPLES - len(
        selected_pairs
    )

    selected_pairs.extend(
        background_pairs[:remaining]
    )


print()
print(
    f"Visualizations to generate : "
    f"{len(selected_pairs)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


def dice_coefficient(y_true, y_pred, smooth=1e-6):

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32
    )

    y_true_flat = tf.reshape(
        y_true,
        [tf.shape(y_true)[0], -1]
    )

    y_pred_flat = tf.reshape(
        y_pred,
        [tf.shape(y_pred)[0], -1]
    )

    intersection = tf.reduce_sum(
        y_true_flat * y_pred_flat,
        axis=1
    )

    denominator = (
        tf.reduce_sum(
            y_true_flat,
            axis=1
        )
        +
        tf.reduce_sum(
            y_pred_flat,
            axis=1
        )
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):

    return 1.0 - dice_coefficient(
        y_true,
        y_pred
    )


def combined_loss(y_true, y_pred):

    bce = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            y_true,
            y_pred
        )
    )

    return bce + dice_loss(
        y_true,
        y_pred
    )


def iou_metric(y_true, y_pred, smooth=1e-6):

    y_true = tf.cast(
        y_true > 0.5,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred > 0.5,
        tf.float32
    )

    y_true_flat = tf.reshape(
        y_true,
        [tf.shape(y_true)[0], -1]
    )

    y_pred_flat = tf.reshape(
        y_pred,
        [tf.shape(y_pred)[0], -1]
    )

    intersection = tf.reduce_sum(
        y_true_flat * y_pred_flat,
        axis=1
    )

    union = (
        tf.reduce_sum(
            y_true_flat,
            axis=1
        )
        +
        tf.reduce_sum(
            y_pred_flat,
            axis=1
        )
        -
        intersection
    )

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return tf.reduce_mean(iou)


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "combined_loss": combined_loss,
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "iou_metric": iou_metric
    }
)

print()
print("MODEL LOAD SUCCESS")
print(
    f"Model parameters : {model.count_params():,}"
)


# ============================================================
# GENERATE VISUALIZATIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING VISUALIZATIONS")
print("=" * 70)


summary = []


for index, (image_path, mask_path) in enumerate(
    selected_pairs,
    start=1
):

    # --------------------------------------------------------
    # Load CT image
    # --------------------------------------------------------

    image = tf.io.read_file(
        str(image_path)
    )

    image = tf.image.decode_png(
        image,
        channels=1
    )

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method=tf.image.ResizeMethod.BILINEAR
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = image / 255.0


    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    mask = tf.io.read_file(
        str(mask_path)
    )

    mask = tf.image.decode_png(
        mask,
        channels=1
    )

    mask = tf.image.resize(
        mask,
        [IMAGE_SIZE, IMAGE_SIZE],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    mask = tf.cast(
        mask,
        tf.float32
    )

    mask = mask / 255.0

    mask = tf.where(
        mask >= 0.5,
        1.0,
        0.0
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    input_tensor = tf.expand_dims(
        image,
        axis=0
    )

    prediction_probability = model.predict(
        input_tensor,
        verbose=0
    )[0, :, :, 0]

    prediction = (
        prediction_probability
        >= PREDICTION_THRESHOLD
    ).astype(np.float32)


    # --------------------------------------------------------
    # Convert arrays
    # --------------------------------------------------------

    image_array = image.numpy()[:, :, 0]

    mask_array = mask.numpy()[:, :, 0]


    # --------------------------------------------------------
    # Calculate per-image Dice and IoU
    # --------------------------------------------------------

    intersection = np.sum(
        mask_array * prediction
    )

    true_area = np.sum(mask_array)
    predicted_area = np.sum(prediction)

    union = (
        true_area
        +
        predicted_area
        -
        intersection
    )

    dice = (
        (2.0 * intersection)
        /
        (true_area + predicted_area + 1e-6)
    )

    iou = (
        intersection
        /
        (union + 1e-6)
    )


    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = np.stack(
        [
            image_array,
            image_array,
            image_array
        ],
        axis=-1
    )

    # Ground truth shown as red.
    overlay[
        mask_array > 0,
        0
    ] = 1.0

    overlay[
        mask_array > 0,
        1
    ] *= 0.35

    overlay[
        mask_array > 0,
        2
    ] *= 0.35

    # Prediction shown as green.
    overlay[
        prediction > 0,
        1
    ] = 1.0

    overlay[
        prediction > 0,
        0
    ] *= 0.35

    overlay[
        prediction > 0,
        2
    ] *= 0.35


    # --------------------------------------------------------
    # Create four-panel figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    axes[0].imshow(
        image_array,
        cmap="gray"
    )

    axes[0].set_title(
        "CT Image"
    )

    axes[1].imshow(
        mask_array,
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[2].imshow(
        prediction,
        cmap="gray"
    )

    axes[2].set_title(
        "U-Net Prediction"
    )

    axes[3].imshow(
        overlay
    )

    axes[3].set_title(
        f"Overlay\nDice={dice:.3f}, IoU={iou:.3f}"
    )


    for ax in axes:
        ax.axis("off")


    fig.suptitle(
        f"Adrenal Tumor Segmentation - Test Sample {index}",
        fontsize=14
    )

    fig.tight_layout()


    # --------------------------------------------------------
    # Save figure
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"test_sample_{index:02d}.png"
    )

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary.append(
        {
            "sample": index,
            "image": image_path.name,
            "mask": mask_path.name,
            "dice": float(dice),
            "iou": float(iou),
            "ground_truth_pixels": int(true_area),
            "predicted_pixels": int(predicted_area)
        }
    )


    print(
        f"{index:02d}. "
        f"{image_path.name} | "
        f"Dice={dice:.4f} | "
        f"IoU={iou:.4f}"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_file = (
    OUTPUT_DIR
    / "visualization_summary.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    import json

    json.dump(
        summary,
        f,
        indent=4
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("STEP 13 COMPLETE")
print("=" * 70)

print(
    f"Visualizations generated : "
    f"{len(summary)}"
)

print(
    f"Output directory         : "
    f"{OUTPUT_DIR}"
)

print(
    f"Summary                  : "
    f"{summary_file}"
)

print()
print(
    "STEP 13 STATUS: "
    "SEGMENTATION VISUALIZATIONS GENERATED SUCCESSFULLY"
)

print("=" * 70)