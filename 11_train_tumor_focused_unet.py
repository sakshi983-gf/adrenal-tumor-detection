from pathlib import Path
import json
import random

import numpy as np
import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers, models
# noinspection PyUnresolvedReferences
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
)
import matplotlib.pyplot as plt


# ============================================================
# STEP 11 - TRAIN U-NET
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = PROJECT_ROOT / "dataset" / "unet_dataset"

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "dataset" / "metadata" / "step11"

BEST_MODEL_FILE = MODEL_DIR / "unet_adrenal_tumor_best.keras"
FINAL_MODEL_FILE = MODEL_DIR / "unet_adrenal_tumor_final.keras"

HISTORY_FILE = OUTPUT_DIR / "training_history.json"
REPORT_FILE = OUTPUT_DIR / "training_report.json"
LOG_FILE = OUTPUT_DIR / "training_log.csv"

LOSS_PLOT = OUTPUT_DIR / "training_loss.png"
DICE_PLOT = OUTPUT_DIR / "training_dice.png"
IOU_PLOT = OUTPUT_DIR / "training_iou.png"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 1

BATCH_SIZE = 2

EPOCHS = 50

LEARNING_RATE = 1e-4

SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# GPU / CPU INFORMATION
# ============================================================

print("=" * 70)
print("STEP 11 - TRAIN U-NET")
print("=" * 70)

print()
print(f"TensorFlow version : {tf.__version__}")

gpus = tf.config.list_physical_devices("GPU")

if gpus:

    print(f"GPU devices        : {len(gpus)}")

    for gpu in gpus:
        print(f"  {gpu}")

else:

    print("GPU devices        : 0")
    print("Training device    : CPU")

print()


# ============================================================
# DIRECTORIES
# ============================================================

TRAIN_IMAGE_DIR = DATASET_ROOT / "train" / "images"
TRAIN_MASK_DIR = DATASET_ROOT / "train" / "masks"

VAL_IMAGE_DIR = DATASET_ROOT / "val" / "images"
VAL_MASK_DIR = DATASET_ROOT / "val" / "masks"


required_dirs = [
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    VAL_IMAGE_DIR,
    VAL_MASK_DIR,
]

for directory in required_dirs:

    if not directory.exists():

        raise FileNotFoundError(
            f"Required directory not found:\n{directory}"
        )


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# IMAGE / MASK PAIRS
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

    if not common_names:

        raise RuntimeError(
            f"No image/mask pairs found:\n{image_dir}"
        )

    if len(image_map) != len(mask_map):

        raise RuntimeError(
            f"Image/mask count mismatch:\n"
            f"{image_dir}\n"
            f"{mask_dir}"
        )

    return (
        [image_map[name] for name in common_names],
        [mask_map[name] for name in common_names],
    )


train_images, train_masks = get_pairs(
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR
)

val_images, val_masks = get_pairs(
    VAL_IMAGE_DIR,
    VAL_MASK_DIR
)


print("Training dataset")
print("-" * 70)

print(
    f"Training samples   : {len(train_images)}"
)

print(
    f"Validation samples : {len(val_images)}"
)

print()


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image = tf.io.read_file(
        image_path
    )

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
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS
        ]
    )

    return image


# ============================================================
# LOAD MASK
# ============================================================

def load_mask(mask_path):

    mask = tf.io.read_file(
        mask_path
    )

    mask = tf.image.decode_png(
        mask,
        channels=1
    )

    mask = tf.image.convert_image_dtype(
        mask,
        tf.float32
    )

    mask = tf.cast(
        mask > 0.5,
        tf.float32
    )

    mask = tf.ensure_shape(
        mask,
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            1
        ]
    )

    return mask


# ============================================================
# PAIRED AUGMENTATION
# ============================================================

def augment(image, mask):

    # Horizontal flip
    random_value = tf.random.uniform(())

    image, mask = tf.cond(
        random_value < 0.5,

        lambda: (
            tf.image.flip_left_right(image),
            tf.image.flip_left_right(mask)
        ),

        lambda: (
            image,
            mask
        )
    )


    # Vertical flip
    random_value = tf.random.uniform(())

    image, mask = tf.cond(
        random_value < 0.5,

        lambda: (
            tf.image.flip_up_down(image),
            tf.image.flip_up_down(mask)
        ),

        lambda: (
            image,
            mask
        )
    )


    return image, mask


# ============================================================
# DATASET
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

        image = load_image(
            image_path
        )

        mask = load_mask(
            mask_path
        )

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
        BATCH_SIZE
    )

    dataset = dataset.prefetch(
        AUTOTUNE
    )

    return dataset


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


# ============================================================
# DICE COEFFICIENT
# ============================================================

def dice_coefficient(
    y_true,
    y_pred
):

    smooth = 1e-6

    y_true = tf.cast(
        y_true,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred,
        tf.float32
    )

    y_true = tf.reshape(
        y_true,
        [tf.shape(y_true)[0], -1]
    )

    y_pred = tf.reshape(
        y_pred,
        [tf.shape(y_pred)[0], -1]
    )

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=1
    )

    denominator = (
        tf.reduce_sum(
            y_true,
            axis=1
        )
        +
        tf.reduce_sum(
            y_pred,
            axis=1
        )
    )

    dice = (
        (2.0 * intersection + smooth)
        /
        (denominator + smooth)
    )

    return tf.reduce_mean(dice)


# ============================================================
# DICE LOSS
# ============================================================

def dice_loss(
    y_true,
    y_pred
):

    return 1.0 - dice_coefficient(
        y_true,
        y_pred
    )


# ============================================================
# COMBINED LOSS
# ============================================================

def combined_loss(
    y_true,
    y_pred
):

    bce = tf.keras.backend.binary_crossentropy(
        y_true,
        y_pred
    )

    bce = tf.reduce_mean(
        bce
    )

    return bce + dice_loss(
        y_true,
        y_pred
    )


# ============================================================
# IOU
# ============================================================

def iou_metric(
    y_true,
    y_pred
):

    smooth = 1e-6

    y_true = tf.cast(
        y_true > 0.5,
        tf.float32
    )

    y_pred = tf.cast(
        y_pred > 0.5,
        tf.float32
    )

    intersection = tf.reduce_sum(
        y_true * y_pred,
        axis=[1, 2, 3]
    )

    union = (
        tf.reduce_sum(
            y_true,
            axis=[1, 2, 3]
        )
        +
        tf.reduce_sum(
            y_pred,
            axis=[1, 2, 3]
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


# ============================================================
# U-NET BLOCK
# ============================================================

def conv_block(
    inputs,
    filters
):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation(
        "relu"
    )(x)

    return x


# ============================================================
# BUILD U-NET
# ============================================================

def build_unet():

    inputs = layers.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS
        ),
        name="ct_input"
    )


    # Encoder
    c1 = conv_block(
        inputs,
        32
    )

    p1 = layers.MaxPooling2D(
        (2, 2)
    )(c1)


    c2 = conv_block(
        p1,
        64
    )

    p2 = layers.MaxPooling2D(
        (2, 2)
    )(c2)


    c3 = conv_block(
        p2,
        128
    )

    p3 = layers.MaxPooling2D(
        (2, 2)
    )(c3)


    c4 = conv_block(
        p3,
        256
    )

    p4 = layers.MaxPooling2D(
        (2, 2)
    )(c4)


    # Bottleneck
    c5 = conv_block(
        p4,
        512
    )


    # Decoder
    u6 = layers.Conv2DTranspose(
        256,
        2,
        strides=(2, 2),
        padding="same"
    )(c5)

    u6 = layers.Concatenate()(
        [u6, c4]
    )

    c6 = conv_block(
        u6,
        256
    )


    u7 = layers.Conv2DTranspose(
        128,
        2,
        strides=(2, 2),
        padding="same"
    )(c6)

    u7 = layers.Concatenate()(
        [u7, c3]
    )

    c7 = conv_block(
        u7,
        128
    )


    u8 = layers.Conv2DTranspose(
        64,
        2,
        strides=(2, 2),
        padding="same"
    )(c7)

    u8 = layers.Concatenate()(
        [u8, c2]
    )

    c8 = conv_block(
        u8,
        64
    )


    u9 = layers.Conv2DTranspose(
        32,
        2,
        strides=(2, 2),
        padding="same"
    )(c8)

    u9 = layers.Concatenate()(
        [u9, c1]
    )

    c9 = conv_block(
        u9,
        32
    )


    outputs = layers.Conv2D(
        1,
        1,
        activation="sigmoid",
        name="tumor_output"
    )(c9)


    return models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Adrenal_Tumor_UNet"
    )


# ============================================================
# CREATE MODEL
# ============================================================

print("=" * 70)
print("Creating U-Net")
print("=" * 70)

model = build_unet()


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=combined_loss,

    metrics=[
        dice_coefficient,
        iou_metric,
        "accuracy",
    ]
)


print()
print(
    f"Model parameters : "
    f"{model.count_params():,}"
)

print(
    f"Learning rate     : "
    f"{LEARNING_RATE}"
)

print(
    f"Loss              : "
    f"Dice Loss + Binary Cross-Entropy"
)

print()


# ============================================================
# CALLBACKS
# ============================================================

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_FILE,
    monitor="val_dice_coefficient",
    mode="max",
    save_best_only=True,
    verbose=1
)


early_stopping = EarlyStopping(
    monitor="val_dice_coefficient",
    mode="max",
    patience=10,
    restore_best_weights=True,
    verbose=1
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_dice_coefficient",
    mode="max",
    factor=0.5,
    patience=4,
    min_lr=1e-7,
    verbose=1
)


csv_logger = CSVLogger(
    LOG_FILE,
    append=False
)


callbacks = [
    checkpoint,
    early_stopping,
    reduce_lr,
    csv_logger,
]


# ============================================================
# TRAIN
# ============================================================

print("=" * 70)
print("STARTING U-NET TRAINING")
print("=" * 70)

print()
print(f"Epochs             : {EPOCHS}")
print(f"Batch size         : {BATCH_SIZE}")
print(f"Training samples   : {len(train_images)}")
print(f"Validation samples : {len(val_images)}")
print()


history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

model.save(
    FINAL_MODEL_FILE
)


# ============================================================
# HISTORY
# ============================================================

history_data = {}

for key, values in history.history.items():

    history_data[key] = [
        float(value)
        for value in values
    ]


with open(
    HISTORY_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        history_data,
        file,
        indent=4
    )


# ============================================================
# BEST EPOCH
# ============================================================

val_dice_values = history_data.get(
    "val_dice_coefficient",
    []
)

if val_dice_values:

    best_epoch_index = int(
        np.argmax(
            val_dice_values
        )
    )

    best_epoch = (
        best_epoch_index + 1
    )

    best_val_dice = float(
        val_dice_values[
            best_epoch_index
        ]
    )

else:

    best_epoch = None
    best_val_dice = None


# ============================================================
# TRAINING REPORT
# ============================================================

report = {

    "tensorflow_version":
        tf.__version__,

    "model":
        "Adrenal_Tumor_UNet",

    "architecture":
        "U-Net",

    "input_shape":
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS
        ],

    "training_samples":
        len(train_images),

    "validation_samples":
        len(val_images),

    "batch_size":
        BATCH_SIZE,

    "maximum_epochs":
        EPOCHS,

    "actual_epochs":
        len(
            history_data.get(
                "loss",
                []
            )
        ),

    "learning_rate":
        LEARNING_RATE,

    "loss":
        "Binary Cross-Entropy + Dice Loss",

    "metrics":
        [
            "dice_coefficient",
            "iou_metric",
            "accuracy"
        ],

    "augmentation":
        [
            "horizontal_flip",
            "vertical_flip"
        ],

    "best_epoch":
        best_epoch,

    "best_validation_dice":
        best_val_dice,

    "best_model":
        str(BEST_MODEL_FILE),

    "final_model":
        str(FINAL_MODEL_FILE),

    "history_file":
        str(HISTORY_FILE),

    "status":
        "PASS"
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
# PLOT TRAINING LOSS
# ============================================================

plt.figure()

plt.plot(
    history_data["loss"],
    label="Training Loss"
)

plt.plot(
    history_data["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("U-Net Training Loss")
plt.legend()
plt.grid(True)

plt.savefig(
    LOSS_PLOT,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT DICE
# ============================================================

plt.figure()

plt.plot(
    history_data["dice_coefficient"],
    label="Training Dice"
)

plt.plot(
    history_data["val_dice_coefficient"],
    label="Validation Dice"
)

plt.xlabel("Epoch")
plt.ylabel("Dice")
plt.title("U-Net Dice Coefficient")
plt.legend()
plt.grid(True)

plt.savefig(
    DICE_PLOT,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# PLOT IOU
# ============================================================

plt.figure()

plt.plot(
    history_data["iou_metric"],
    label="Training IoU"
)

plt.plot(
    history_data["val_iou_metric"],
    label="Validation IoU"
)

plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("U-Net IoU")
plt.legend()
plt.grid(True)

plt.savefig(
    IOU_PLOT,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("STEP 11 COMPLETE")
print("=" * 70)

print(
    f"Training epochs     : "
    f"{len(history_data.get('loss', []))}"
)

print(
    f"Best epoch          : "
    f"{best_epoch}"
)

if best_val_dice is not None:

    print(
        f"Best validation Dice : "
        f"{best_val_dice:.6f}"
    )

print()
print(
    f"Best model          : "
    f"{BEST_MODEL_FILE}"
)

print(
    f"Final model         : "
    f"{FINAL_MODEL_FILE}"
)

print(
    f"Training history    : "
    f"{HISTORY_FILE}"
)

print(
    f"Training report     : "
    f"{REPORT_FILE}"
)

print(
    f"Training log        : "
    f"{LOG_FILE}"
)

print()
print(
    "Training curves saved:"
)

print(
    f"  Loss : {LOSS_PLOT}"
)

print(
    f"  Dice : {DICE_PLOT}"
)

print(
    f"  IoU  : {IOU_PLOT}"
)

print()
print(
    "STEP 11 STATUS: "
    "U-NET TRAINING COMPLETED SUCCESSFULLY"
)