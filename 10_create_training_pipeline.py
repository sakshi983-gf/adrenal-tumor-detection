from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf


# ============================================================
# STEP 10 - CREATE TRAINING DATA PIPELINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "dataset" / "unet_dataset"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "metadata" / "step10"
REPORT_FILE = OUTPUT_DIR / "training_pipeline_report.json"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 512
CHANNELS = 1

BATCH_SIZE = 2

SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# DIRECTORIES
# ============================================================

TRAIN_IMAGE_DIR = DATASET_ROOT / "train" / "images"
TRAIN_MASK_DIR = DATASET_ROOT / "train" / "masks"

VAL_IMAGE_DIR = DATASET_ROOT / "val" / "images"
VAL_MASK_DIR = DATASET_ROOT / "val" / "masks"

TEST_IMAGE_DIR = DATASET_ROOT / "test" / "images"
TEST_MASK_DIR = DATASET_ROOT / "test" / "masks"


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 10 - CREATE TRAINING DATA PIPELINE")
print("=" * 70)

print()
print(f"TensorFlow version : {tf.__version__}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Image size         : {IMAGE_SIZE} x {IMAGE_SIZE}")
print()


# ============================================================
# CHECK DIRECTORIES
# ============================================================

required_directories = [
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
    TEST_IMAGE_DIR,
    TEST_MASK_DIR,
]

for directory in required_directories:

    if not directory.exists():

        raise FileNotFoundError(
            f"Required directory not found:\n{directory}"
        )


# ============================================================
# GET IMAGE / MASK PAIRS
# ============================================================

def get_pairs(image_dir, mask_dir):

    image_files = sorted(
        image_dir.glob("*.png")
    )

    mask_files = sorted(
        mask_dir.glob("*.png")
    )

    image_map = {
        file.name: str(file)
        for file in image_files
    }

    mask_map = {
        file.name: str(file)
        for file in mask_files
    }

    common_names = sorted(
        set(image_map.keys())
        & set(mask_map.keys())
    )

    if len(common_names) == 0:

        raise RuntimeError(
            f"No image/mask pairs found in:\n{image_dir}"
        )

    missing_masks = sorted(
        set(image_map.keys())
        - set(mask_map.keys())
    )

    missing_images = sorted(
        set(mask_map.keys())
        - set(image_map.keys())
    )

    if missing_masks:

        raise RuntimeError(
            f"Missing masks in {mask_dir}: "
            f"{len(missing_masks)}"
        )

    if missing_images:

        raise RuntimeError(
            f"Missing images in {image_dir}: "
            f"{len(missing_images)}"
        )

    return (
        [image_map[name] for name in common_names],
        [mask_map[name] for name in common_names],
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_png(
        image,
        channels=1
    )

    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    image = tf.ensure_shape(
        image,
        [IMAGE_SIZE, IMAGE_SIZE, CHANNELS]
    )

    return image


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path):

    mask = tf.io.read_file(mask_path)

    mask = tf.image.decode_png(
        mask,
        channels=1
    )

    mask = tf.image.convert_image_dtype(
        mask,
        tf.float32
    )

    # Convert 0/1 floating mask explicitly
    mask = tf.cast(
        mask > 0.5,
        tf.float32
    )

    mask = tf.ensure_shape(
        mask,
        [IMAGE_SIZE, IMAGE_SIZE, 1]
    )

    return mask


# ============================================================
# PAIRED AUGMENTATION
# ============================================================

def augment(image, mask):

    # Horizontal flip
    random_value = tf.random.uniform(())

    def flip():

        return (
            tf.image.flip_left_right(image),
            tf.image.flip_left_right(mask)
        )

    image, mask = tf.cond(
        random_value < 0.5,
        flip,
        lambda: (image, mask)
    )


    # Vertical flip
    random_value = tf.random.uniform(())

    def flip():

        return (
            tf.image.flip_up_down(image),
            tf.image.flip_up_down(mask)
        )

    image, mask = tf.cond(
        random_value < 0.5,
        flip,
        lambda: (image, mask)
    )


    return image, mask


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset(
    image_paths,
    mask_paths,
    training=False
):

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            mask_paths
        )
    )

    if training:

        dataset = dataset.shuffle(
            buffer_size=len(image_paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    def load_pair(
        image_path,
        mask_path
    ):

        image = load_image(image_path)

        mask = load_mask(mask_path)

        return image, mask


    dataset = dataset.map(
        load_pair,
        num_parallel_calls=AUTOTUNE
    )

    if training:

        dataset = dataset.map(
            augment,
            num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


# ============================================================
# CREATE TRAIN / VAL / TEST PAIRS
# ============================================================

train_images, train_masks = get_pairs(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR
)

val_images, val_masks = get_pairs(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
)

test_images, test_masks = get_pairs(
    TEST_IMAGE_DIR,
    TEST_MASK_DIR
)


print("Dataset pairs")
print("-" * 70)

print(
    f"Training pairs   : {len(train_images)}"
)

print(
    f"Validation pairs : {len(val_images)}"
)

print(
    f"Test pairs       : {len(test_images)}"
)

print()


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = create_dataset(
    train_images,
    train_masks,
    training=True
)

val_dataset = create_dataset(
    val_images,
    val_masks,
    training=False
)

test_dataset = create_dataset(
    test_images,
    test_masks,
    training=False
)


# ============================================================
# VERIFY ONE TRAINING BATCH
# ============================================================

print("-" * 70)
print("Verifying training batch")
print("-" * 70)

train_batch_images, train_batch_masks = next(
    iter(train_dataset)
)


print(
    f"Image batch shape : "
    f"{train_batch_images.shape}"
)

print(
    f"Mask batch shape  : "
    f"{train_batch_masks.shape}"
)


# ============================================================
# VERIFY ONE VALIDATION BATCH
# ============================================================

val_batch_images, val_batch_masks = next(
    iter(val_dataset)
)


print(
    f"Val image shape   : "
    f"{val_batch_images.shape}"
)

print(
    f"Val mask shape    : "
    f"{val_batch_masks.shape}"
)


# ============================================================
# VERIFY ONE TEST BATCH
# ============================================================

test_batch_images, test_batch_masks = next(
    iter(test_dataset)
)


print(
    f"Test image shape  : "
    f"{test_batch_images.shape}"
)


print(
    f"Test mask shape   : "
    f"{test_batch_masks.shape}"
)


# ============================================================
# NUMERICAL VALIDATION
# ============================================================

image_min = float(
    tf.reduce_min(train_batch_images).numpy()
)

image_max = float(
    tf.reduce_max(train_batch_images).numpy()
)

mask_min = float(
    tf.reduce_min(train_batch_masks).numpy()
)

mask_max = float(
    tf.reduce_max(train_batch_masks).numpy()
)


print()
print("-" * 70)
print("Numerical validation")
print("-" * 70)

print(
    f"Image range : "
    f"{image_min:.6f} - {image_max:.6f}"
)

print(
    f"Mask range  : "
    f"{mask_min:.6f} - {mask_max:.6f}"
)


# ============================================================
# CHECK VALUES
# ============================================================

if image_min < 0.0 or image_max > 1.0:

    raise RuntimeError(
        "Image normalization failed."
    )


if mask_min < 0.0 or mask_max > 1.0:

    raise RuntimeError(
        "Mask normalization failed."
    )


# ============================================================
# CHECK MASK BINARY
# ============================================================

unique_mask_values = np.unique(
    train_batch_masks.numpy()
)

print(
    f"Mask unique values : "
    f"{unique_mask_values.tolist()}"
)


allowed_values = {0.0, 1.0}

if not set(
    unique_mask_values.tolist()
).issubset(allowed_values):

    raise RuntimeError(
        "Mask contains values other than 0 and 1."
    )


# ============================================================
# CHECK IMAGE / MASK SHAPES
# ============================================================

if train_batch_images.shape != train_batch_masks.shape:

    raise RuntimeError(
        "Training image/mask batch shapes do not match."
    )


if val_batch_images.shape != val_batch_masks.shape:

    raise RuntimeError(
        "Validation image/mask batch shapes do not match."
    )


if test_batch_images.shape != test_batch_masks.shape:

    raise RuntimeError(
        "Test image/mask batch shapes do not match."
    )


# ============================================================
# COUNT BATCHES
# ============================================================

train_batch_count = int(
    tf.data.experimental.cardinality(
        train_dataset
    ).numpy()
)

val_batch_count = int(
    tf.data.experimental.cardinality(
        val_dataset
    ).numpy()
)

test_batch_count = int(
    tf.data.experimental.cardinality(
        test_dataset
    ).numpy()
)


print()
print("-" * 70)
print("Batch information")
print("-" * 70)

print(
    f"Training batches   : {train_batch_count}"
)

print(
    f"Validation batches : {val_batch_count}"
)

print(
    f"Test batches       : {test_batch_count}"
)


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


report = {

    "tensorflow_version":
        tf.__version__,

    "image_size":
        [IMAGE_SIZE, IMAGE_SIZE, CHANNELS],

    "batch_size":
        BATCH_SIZE,

    "seed":
        SEED,

    "train_samples":
        len(train_images),

    "validation_samples":
        len(val_images),

    "test_samples":
        len(test_images),

    "train_batches":
        train_batch_count,

    "validation_batches":
        val_batch_count,

    "test_batches":
        test_batch_count,

    "image_normalization":
        "0-255 to 0-1",

    "mask_normalization":
        "0/255 to binary 0/1",

    "augmentation":
        {
            "horizontal_flip": True,
            "vertical_flip": True,
        },

    "train_image_batch_shape":
        list(train_batch_images.shape),

    "train_mask_batch_shape":
        list(train_batch_masks.shape),

    "validation_image_batch_shape":
        list(val_batch_images.shape),

    "validation_mask_batch_shape":
        list(val_batch_masks.shape),

    "test_image_batch_shape":
        list(test_batch_images.shape),

    "test_mask_batch_shape":
        list(test_batch_masks.shape),

    "image_min":
        image_min,

    "image_max":
        image_max,

    "mask_min":
        mask_min,

    "mask_max":
        mask_max,

    "mask_unique_values":
        unique_mask_values.tolist(),

    "status":
        "PASS",
}


with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        report,
        file,
        indent=4
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("STEP 10 COMPLETE")
print("=" * 70)

print(
    f"Training samples   : {len(train_images)}"
)

print(
    f"Validation samples : {len(val_images)}"
)

print(
    f"Test samples       : {len(test_images)}"
)

print(
    f"Image range        : "
    f"{image_min:.6f} - {image_max:.6f}"
)

print(
    f"Mask range         : "
    f"{mask_min:.6f} - {mask_max:.6f}"
)

print()
print(
    f"Report : {REPORT_FILE}"
)

print()

print(
    "STEP 10 STATUS: "
    "TRAINING DATA PIPELINE VERIFIED SUCCESSFULLY"
)