from pathlib import Path
import json
import tensorflow as tf
# noinspection PyUnresolvedReferences
from tensorflow.keras import layers, models


# ============================================================
# STEP 09 - CREATE U-NET MODEL
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models"
METADATA_DIR = PROJECT_ROOT / "dataset" / "metadata" / "step09"

MODEL_FILE = MODEL_DIR / "unet_adrenal_tumor.keras"
SUMMARY_FILE = METADATA_DIR / "unet_model_summary.txt"
REPORT_FILE = METADATA_DIR / "unet_model_report.json"


# ============================================================
# SETTINGS
# ============================================================

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
IMAGE_CHANNELS = 1

NUM_CLASSES = 1


# ============================================================
# U-NET BUILDING BLOCK
# ============================================================

def conv_block(inputs, filters):

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal"
    )(inputs)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    x = layers.Conv2D(
        filters,
        3,
        padding="same",
        kernel_initializer="he_normal"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)

    return x


# ============================================================
# CREATE U-NET
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


    # --------------------------------------------------------
    # ENCODER
    # --------------------------------------------------------

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(c1)


    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(c2)


    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(c3)


    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D(
        pool_size=(2, 2)
    )(c4)


    # --------------------------------------------------------
    # BOTTLENECK
    # --------------------------------------------------------

    c5 = conv_block(p4, 512)


    # --------------------------------------------------------
    # DECODER
    # --------------------------------------------------------

    u6 = layers.Conv2DTranspose(
        256,
        2,
        strides=(2, 2),
        padding="same"
    )(c5)

    u6 = layers.Concatenate()(
        [u6, c4]
    )

    c6 = conv_block(u6, 256)


    u7 = layers.Conv2DTranspose(
        128,
        2,
        strides=(2, 2),
        padding="same"
    )(c6)

    u7 = layers.Concatenate()(
        [u7, c3]
    )

    c7 = conv_block(u7, 128)


    u8 = layers.Conv2DTranspose(
        64,
        2,
        strides=(2, 2),
        padding="same"
    )(c7)

    u8 = layers.Concatenate()(
        [u8, c2]
    )

    c8 = conv_block(u8, 64)


    u9 = layers.Conv2DTranspose(
        32,
        2,
        strides=(2, 2),
        padding="same"
    )(c8)

    u9 = layers.Concatenate()(
        [u9, c1]
    )

    c9 = conv_block(u9, 32)


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    outputs = layers.Conv2D(
        NUM_CLASSES,
        1,
        activation="sigmoid",
        name="tumor_output"
    )(c9)


    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="Adrenal_Tumor_UNet"
    )

    return model


# ============================================================
# CREATE MODEL
# ============================================================

print("=" * 70)
print("STEP 09 - CREATE U-NET MODEL")
print("=" * 70)

print()
print(f"TensorFlow version : {tf.__version__}")
print()


model = build_unet()


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-4
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy"
    ]
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METADATA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_FILE
)


# ============================================================
# SAVE MODEL SUMMARY
# ============================================================

with open(
    SUMMARY_FILE,
    "w",
    encoding="utf-8"
) as file:

    model.summary(
        print_fn=file.write
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = model.count_params()

trainable_parameters = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.trainable_weights
)

non_trainable_parameters = sum(
    tf.keras.backend.count_params(weight)
    for weight in model.non_trainable_weights
)


report = {

    "model_name":
        "Adrenal_Tumor_UNet",

    "architecture":
        "U-Net",

    "task":
        "Binary adrenal tumor segmentation",

    "input_shape":
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            IMAGE_CHANNELS
        ],

    "output_shape":
        [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            NUM_CLASSES
        ],

    "activation":
        "sigmoid",

    "optimizer":
        "Adam",

    "learning_rate":
        0.0001,

    "loss":
        "binary_crossentropy",

    "total_parameters":
        int(total_parameters),

    "trainable_parameters":
        int(trainable_parameters),

    "non_trainable_parameters":
        int(non_trainable_parameters),

    "tensorflow_version":
        tf.__version__,

    "model_file":
        str(MODEL_FILE),
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
# FINAL OUTPUT
# ============================================================

print("=" * 70)
print("STEP 09 COMPLETE")
print("=" * 70)

print(
    f"Model             : {model.name}"
)

print(
    f"Input shape       : {model.input_shape}"
)

print(
    f"Output shape      : {model.output_shape}"
)

print(
    f"Total parameters  : {total_parameters:,}"
)

print(
    f"Trainable params  : {trainable_parameters:,}"
)

print(
    f"Non-trainable     : {non_trainable_parameters:,}"
)

print()
print(
    f"Model saved       : {MODEL_FILE}"
)

print(
    f"Summary saved     : {SUMMARY_FILE}"
)

print(
    f"Report saved      : {REPORT_FILE}"
)

print()

print(
    "STEP 09 STATUS: U-NET MODEL CREATED SUCCESSFULLY"
)
