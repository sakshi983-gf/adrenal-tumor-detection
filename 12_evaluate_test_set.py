# ======================================================================
# STEP 12 - FINAL TEST SET EVALUATION
# STACKED: 8-View TTA + RankSEG + 2D CCA + Per-Patient 3D CCA
# ======================================================================
# Key fix: 3D CCA is applied PER PATIENT, not across the whole test set.
# Patient IDs are extracted from filenames like:
#   Adrenal_Ki67_Seg_048_slice_054.png  ->  patient = 048
# ======================================================================

from pathlib import Path
import re
import json
import numpy as np
import tensorflow as tf
from scipy import ndimage

# RankSEG imports (optional)
try:
    import torch
    from rankseg import RankSEG
    RANKSEG_AVAILABLE = True
except ImportError:
    RANKSEG_AVAILABLE = False
    print("\n[WARNING] rankseg/torch not installed. RankSEG step will be skipped.")


# ======================================================================
# CONFIGURATION
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset" / "unet_dataset"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tumor_focused_resume_training_v2"
    / "best_model.keras"
)

TEST_IMAGE_DIR = DATASET_ROOT / "test" / "images"
TEST_MASK_DIR  = DATASET_ROOT / "test" / "masks"

OUTPUT_DIR = PROJECT_ROOT / "dataset" / "metadata" / "step12"
REPORT_FILE = OUTPUT_DIR / "test_evaluation_per_patient_3dcca.json"

IMAGE_SIZE = 512
BATCH_SIZE = 2
FIXED_THRESHOLD = 0.80
MIN_COMPONENT_SIZE = 150
EPSILON = 1e-7

PATIENT_ID_PATTERN = re.compile(r"(Adrenal_Ki67_Seg_\d+)")


# ======================================================================
# HEADER
# ======================================================================

print("=" * 70)
print("STEP 12 - FINAL TEST SET EVALUATION")
print("STACKED: 8-View TTA + RankSEG + 2D CCA + Per-Patient 3D CCA")
print("=" * 70)
print()
print(f"TensorFlow version : {tf.__version__}")
print(f"RankSEG available  : {RANKSEG_AVAILABLE}")
print(f"Model              : {MODEL_PATH}")


# ======================================================================
# LOAD BEST MODEL
# ======================================================================

print()
print("=" * 70)
print("LOADING BEST MODEL")
print("=" * 70)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    yt = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    yp = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(yt * yp, axis=1)
    denom = tf.reduce_sum(yt, axis=1) + tf.reduce_sum(yp, axis=1)
    return tf.reduce_mean((2.0 * inter + smooth) / (denom + smooth))


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def combined_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true > 0.5, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    yt = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    yp = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])
    inter = tf.reduce_sum(yt * yp, axis=1)
    union = tf.reduce_sum(yt, axis=1) + tf.reduce_sum(yp, axis=1) - inter
    return tf.reduce_mean((inter + smooth) / (union + smooth))


model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "combined_loss": combined_loss,
        "dice_coefficient": dice_coefficient,
        "dice_loss": dice_loss,
        "iou_metric": iou_metric,
    },
    compile=False,
)

print(f"\nMODEL LOAD SUCCESS")
print(f"Model parameters : {model.count_params():,}")


# ======================================================================
# LOAD TEST DATA (with patient IDs)
# ======================================================================

print()
print("=" * 70)
print("LOADING TEST DATA")
print("=" * 70)

image_files = sorted(TEST_IMAGE_DIR.glob("*.png"))

image_paths = []
mask_paths = []
patient_ids = []

for image_path in image_files:
    mask_path = TEST_MASK_DIR / image_path.name
    if not mask_path.exists():
        continue

    # Extract patient ID from filename
    match = PATIENT_ID_PATTERN.search(image_path.name)
    if not match:
        continue

    image_paths.append(str(image_path))
    mask_paths.append(str(mask_path))
    patient_ids.append(match.group(1))

patient_ids = np.array(patient_ids)
unique_patients = sorted(np.unique(patient_ids))

print(f"Test images    : {len(image_paths)}")
print(f"Unique patients: {len(unique_patients)}")
print(f"Patients       : {', '.join(unique_patients)}")


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_png(image, channels=1)
    image = tf.image.resize(
        image, [IMAGE_SIZE, IMAGE_SIZE],
        method=tf.image.ResizeMethod.BILINEAR
    )
    return (tf.cast(image, tf.float32) / 255.0).numpy()


def load_mask(path):
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(
        mask, [IMAGE_SIZE, IMAGE_SIZE],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    mask = tf.cast(mask, tf.float32) / 255.0
    return tf.where(mask >= 0.5, 1.0, 0.0).numpy()


all_images = np.stack([load_image(p) for p in image_paths], axis=0)
all_masks  = np.stack([load_mask(p)  for p in mask_paths],  axis=0)

print(f"Image tensor: {all_images.shape}")
print(f"Mask tensor : {all_masks.shape}")


# ======================================================================
# 8-VIEW TTA
# ======================================================================

print()
print("=" * 70)
print("8-VIEW TTA ON TEST SET")
print("=" * 70)

all_preds = []
for flip_b, flip_h, flip_w in [
    (False, False, False),
    (False, False, True),
    (False, True,  False),
    (False, True,  True),
    (True,  False, False),
    (True,  False, True),
    (True,  True,  False),
    (True,  True,  True),
]:
    axes = []
    if flip_b: axes.append(0)
    if flip_h: axes.append(1)
    if flip_w: axes.append(2)

    flipped = np.flip(all_images, axis=tuple(axes)) if axes else all_images.copy()
    pred = model.predict(flipped, batch_size=BATCH_SIZE, verbose=0)
    pred = np.flip(pred, axis=tuple(axes)) if axes else pred
    all_preds.append(pred)

tta_probs = np.mean(all_preds, axis=0)
print("8-view TTA complete.")


# ======================================================================
# METRIC COMPUTATION
# ======================================================================

def compute_metrics(y_true, y_pred_binary):
    yt = y_true.reshape(y_true.shape[0], -1).astype(np.float32)
    yp = y_pred_binary.reshape(y_pred_binary.shape[0], -1).astype(np.float32)

    inter = np.sum(yt * yp, axis=1)
    denom = np.sum(yt, axis=1) + np.sum(yp, axis=1)
    union = denom - inter

    dice = (2.0 * inter + EPSILON) / (denom + EPSILON)
    iou = (inter + EPSILON) / (union + EPSILON)
    acc = np.sum(yt == yp, axis=1) / yt.shape[1]

    return {
        "dice": float(np.mean(dice)),
        "iou": float(np.mean(iou)),
        "accuracy": float(np.mean(acc)),
    }


# ======================================================================
# BASELINE @ 0.80
# ======================================================================

baseline_binary = (tta_probs >= FIXED_THRESHOLD).astype(np.float32)
baseline_metrics = compute_metrics(all_masks, baseline_binary)

print()
print("=" * 70)
print("BASELINE @ 0.80 (with 8-view TTA)")
print("=" * 70)
print(f"Dice     : {baseline_metrics['dice']:.4f}")
print(f"IoU      : {baseline_metrics['iou']:.4f}")
print(f"Accuracy : {baseline_metrics['accuracy']:.4f}")


# ======================================================================
# RankSEG
# ======================================================================

rankseg_metrics = None
rankseg_binary = None

if RANKSEG_AVAILABLE:
    print()
    print("=" * 70)
    print("RankSEG THRESHOLD OPTIMIZATION")
    print("=" * 70)

    # Convert to (N, 2, H, W) format for RankSEG
    probs_2ch = np.concatenate([1.0 - tta_probs, tta_probs], axis=-1)
    probs_2ch = np.transpose(probs_2ch, (0, 3, 1, 2))

    probs_tensor = torch.from_numpy(probs_2ch).float()
    rankseg = RankSEG(metric="dice", solver="RMA", output_mode="multiclass")

    with torch.no_grad():
        rankseg_preds = rankseg.predict(probs_tensor)

    rankseg_binary = rankseg_preds.numpy().astype(np.float32)
    rankseg_binary = rankseg_binary[..., np.newaxis]

    rankseg_metrics = compute_metrics(all_masks, rankseg_binary)

    print(f"Dice     : {rankseg_metrics['dice']:.4f}")
    print(f"IoU      : {rankseg_metrics['iou']:.4f}")
    print(f"Accuracy : {rankseg_metrics['accuracy']:.4f}")


# ======================================================================
# 2D CCA (on RankSEG output if available, else baseline)
# ======================================================================

source_binary = rankseg_binary if rankseg_binary is not None else baseline_binary
source_name = "RankSEG" if rankseg_binary is not None else "Baseline"

print()
print("=" * 70)
print(f"2D CCA (on {source_name} output)")
print("=" * 70)


def improved_2d_cca(mask_2d, min_size=MIN_COMPONENT_SIZE):
    labeled, num = ndimage.label(mask_2d)
    filtered = np.zeros_like(mask_2d, dtype=bool)

    for cid in range(1, num + 1):
        if int(np.sum(labeled == cid)) >= min_size:
            filtered[labeled == cid] = True

    filtered = ndimage.binary_fill_holes(filtered)

    labeled, num = ndimage.label(filtered)
    if num > 0:
        sizes = ndimage.sum(filtered, labeled, range(1, num + 1))
        largest = int(np.argmax(sizes)) + 1
        filtered = (labeled == largest)

    return filtered


masks_2d_cca = np.zeros_like(source_binary)
for i in range(source_binary.shape[0]):
    mask_2d = source_binary[i, :, :, 0].astype(bool)
    refined = improved_2d_cca(mask_2d)
    masks_2d_cca[i, :, :, 0] = refined.astype(np.float32)

cca2d_metrics = compute_metrics(all_masks, masks_2d_cca)
print(f"Dice     : {cca2d_metrics['dice']:.4f}")
print(f"IoU      : {cca2d_metrics['iou']:.4f}")
print(f"Accuracy : {cca2d_metrics['accuracy']:.4f}")


# ======================================================================
# PER-PATIENT 3D CCA  ← THE FIX
# ======================================================================

print()
print("=" * 70)
print("PER-PATIENT 3D CCA")
print("=" * 70)
print("Grouping slices by patient, then applying 3D CCA per patient...")

final_binary = np.zeros_like(masks_2d_cca)

for patient in unique_patients:
    # Get indices for this patient
    patient_mask = (patient_ids == patient)
    patient_indices = np.where(patient_mask)[0]

    # Extract 2D CCA masks for this patient
    patient_volume = masks_2d_cca[patient_indices, :, :, 0].astype(bool)
    # Shape: (num_slices_for_patient, 512, 512)

    # Apply 3D CCA
    labeled_3d, num_3d = ndimage.label(patient_volume)

    if num_3d == 0:
        continue

    sizes_3d = ndimage.sum(patient_volume, labeled_3d, range(1, num_3d + 1))
    largest_3d = int(np.argmax(sizes_3d)) + 1
    filtered_volume = (labeled_3d == largest_3d)

    # Write back
    final_binary[patient_indices, :, :, 0] = filtered_volume.astype(np.float32)

    print(f"  {patient}: {len(patient_indices)} slices, "
          f"{num_3d} 3D components → kept largest")

final_metrics = compute_metrics(all_masks, final_binary)

print()
print(f"Final Dice     : {final_metrics['dice']:.4f}")
print(f"Final IoU      : {final_metrics['iou']:.4f}")
print(f"Final Accuracy : {final_metrics['accuracy']:.4f}")


# ======================================================================
# COMPARISON TABLE
# ======================================================================

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

configs = [
    ("8-View TTA @ 0.80", baseline_metrics),
]

if rankseg_metrics is not None:
    configs.append(("+ RankSEG", rankseg_metrics))

configs.append(("+ 2D CCA", cca2d_metrics))
configs.append(("+ Per-Patient 3D CCA (Final)", final_metrics))

print(f"{'Configuration':<38}{'Dice':>10}{'IoU':>10}{'Acc':>10}")
print("-" * 70)
for name, metrics in configs:
    print(
        f"{name:<38}"
        f"{metrics['dice']:>10.4f}"
        f"{metrics['iou']:>10.4f}"
        f"{metrics['accuracy']:>10.4f}"
    )

print()
print("-" * 70)
base = configs[0][1]["dice"]
for name, metrics in configs[1:]:
    print(f"Improvement ({name}): {metrics['dice'] - base:+.4f} Dice")

# Compare to previous best
PREVIOUS_BEST = 0.5597
print()
print("-" * 70)
print(f"Previous best Dice     : {PREVIOUS_BEST:.4f}")
print(f"New result Dice        : {final_metrics['dice']:.4f}")
delta_vs_best = final_metrics["dice"] - PREVIOUS_BEST
if delta_vs_best > 0:
    print(f"✅ IMPROVEMENT         : {delta_vs_best:+.4f}")
else:
    print(f"⚠️  No improvement      : {delta_vs_best:+.4f}")


# ======================================================================
# SAVE REPORT
# ======================================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

report = {
    "step": "12_per_patient_3dcca",
    "description": "8-view TTA + RankSEG + 2D CCA + per-patient 3D CCA",
    "model_parameters": model.count_params(),
    "test_images": len(image_paths),
    "test_patients": len(unique_patients),
    "fixed_threshold": FIXED_THRESHOLD,
    "min_component_size": MIN_COMPONENT_SIZE,
    "rankseg_available": RANKSEG_AVAILABLE,
    "previous_best_dice": PREVIOUS_BEST,
    "configurations": {name: metrics for name, metrics in configs},
    "final_result": final_metrics,
    "improvement_vs_previous_best": delta_vs_best,
}

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, default=float)

print()
print(f"Report saved: {REPORT_FILE}")
print()
print("=" * 70)
print("STEP 12 STATUS: EVALUATION COMPLETED")
print("=" * 70)