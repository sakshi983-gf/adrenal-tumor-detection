import os
# Disable buggy oneDNN CPU conv2d stalls on Windows (cuts 512x512 U-Net pass from ~82s to ~0.21s)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hashlib
import json
import re
import shutil
import sqlite3
import threading
import time
from datetime import timedelta
from functools import wraps

import cv2
import numpy as np
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from inference import (
    classify_body_region,
    compute_burden_metrics,
    validate_file_properties,
    validate_image_characteristics,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "adrenalcare-ai-clinical-secret-key-2026")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)

UPLOAD_FOLDER = "uploads"
TEMP_FOLDER = "temp"
STATIC_MASKS_FOLDER = os.path.join("static", "masks")
LAST_BATCH_FOLDER = os.path.join("temp", "last_batch")
MODEL_PATH = "models/tumor_focused_resume_training_v2/best_model.keras"
DB_PATH = "users.db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(STATIC_MASKS_FOLDER, exist_ok=True)
os.makedirs(LAST_BATCH_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250 MB limit for bulk CT series
app.request_class.max_form_parts = 5000


@app.errorhandler(413)
def handle_request_entity_too_large(e):
    return jsonify({"error": "Maximum 1000 slices per analysis"}), 400

PREDICTION_THRESHOLD = 0.80
MIN_COMPONENT_SIZE = 150

PIXEL_SPACING_MM = 0.742
PIXEL_AREA_MM2 = PIXEL_SPACING_MM * PIXEL_SPACING_MM
SLICE_THICKNESS_MM = 1.25


# =============================================================================
# DATABASE & AUTHENTICATION SETUP
# =============================================================================

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                license_number TEXT NOT NULL,
                institution TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Seed default clinician account if not present
        cur = conn.execute("SELECT id FROM users WHERE email = ?", ("jane@hospital.com",))
        if cur.fetchone() is None:
            conn.execute(
                """
                INSERT INTO users (full_name, email, license_number, institution, password_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "Dr. Jane Smith",
                    "jane@hospital.com",
                    "MD-948201",
                    "Memorial Oncology Hospital",
                    generate_password_hash("Password123"),
                ),
            )
    conn.close()


init_db()


def get_current_user():
    if not session.get("user_id"):
        return None
    return {
        "id": session.get("user_id"),
        "full_name": session.get("full_name", "Dr. Jane Smith"),
        "email": session.get("email", "jane@hospital.com"),
        "license_number": session.get("license_number", "MD-948201"),
        "institution": session.get("institution", "Memorial Oncology Hospital"),
    }


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user_id"):
            if request.path.startswith("/api/"):
                return jsonify({"error": "Authentication required", "redirect": "/login"}), 401
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated_function


# =============================================================================
# U-NET MODEL LOADING & IN-MEMORY CACHING
# =============================================================================

model = None
_predict_fn = None
ndimage = None
_backend_lock = threading.Lock()

# Cache probability maps by SHA-256 hash of each CT slice file for fast repeat & threshold re-runs
_prob_cache = {}  # key: file_hash -> (pred_map_512x512, resized_gray_512x512)


def ensure_backend_ready():
    global model, _predict_fn, ndimage
    if _predict_fn is not None and ndimage is not None:
        return

    with _backend_lock:
        if ndimage is None:
            from scipy import ndimage as _ndimage
            ndimage = _ndimage

        if _predict_fn is None:
            import tensorflow as tf
            try:
                tf.config.threading.set_intra_op_parallelism_threads(4)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except RuntimeError:
                pass

            print("\nLoading trained U-Net model...")
            loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            compiled_fn = tf.function(
                lambda inp: loaded_model(inp, training=False),
                reduce_retracing=True,
            )
            dummy = tf.zeros((1, 512, 512, 1), dtype=tf.float32)
            _ = compiled_fn(dummy)

            model = loaded_model
            _predict_fn = compiled_fn
            print("Model loaded successfully.")
            print("Model name:", model.name)
            print("Parameters:", model.count_params())


def _compute_file_hash(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def preprocess_slice(image_path: str):
    """Load grayscale CT slice, resize to 512x512, normalize to [0, 1]."""
    with Image.open(image_path) as raw_img:
        image = raw_img.convert("L")
        if image.size != (512, 512):
            image = image.resize((512, 512))
        resized_gray = np.asarray(image, dtype=np.uint8)

    image_array = resized_gray.astype(np.float32) * (1.0 / 255.0)
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, resized_gray


def postprocess_mask(prediction_map: np.ndarray, threshold: float):
    ensure_backend_ready()
    binary_mask = (prediction_map >= threshold).astype(np.uint8)
    labeled_mask, num_features = ndimage.label(binary_mask)

    if num_features == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0
    valid_labels = np.where(component_sizes >= MIN_COMPONENT_SIZE)[0]

    if valid_labels.size == 0:
        return np.zeros_like(binary_mask, dtype=np.uint8)

    largest_label = valid_labels[np.argmax(component_sizes[valid_labels])]
    cleaned_mask = (labeled_mask == largest_label).astype(np.uint8)
    cleaned_mask = ndimage.binary_fill_holes(cleaned_mask).astype(np.uint8)
    return cleaned_mask


def save_slice_visuals(resized_gray: np.ndarray, binary_mask: np.ndarray, out_dir: str, idx: int):
    """Save original CT, binary mask, and red tumor overlay PNGs for slice idx."""
    orig_name = f"orig_{idx:03d}.png"
    mask_name = f"mask_{idx:03d}.png"
    overlay_name = f"overlay_{idx:03d}.png"

    orig_path = os.path.join(out_dir, orig_name)
    mask_path = os.path.join(out_dir, mask_name)
    overlay_path = os.path.join(out_dir, overlay_name)

    cv2.imwrite(orig_path, resized_gray)
    cv2.imwrite(mask_path, (binary_mask * 255).astype(np.uint8))

    # Build RGB overlay with red translucent fill and crisp contour
    bgr = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)
    if np.any(binary_mask > 0):
        red_layer = np.zeros_like(bgr)
        red_layer[:, :, 2] = 235  # Red in BGR
        red_layer[:, :, 1] = 40
        red_layer[:, :, 0] = 40
        blended = cv2.addWeighted(bgr, 0.58, red_layer, 0.42, 0)
        overlay = np.where(binary_mask[:, :, None] > 0, blended, bgr)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (40, 40, 220), 2)
    else:
        overlay = bgr

    cv2.imwrite(overlay_path, overlay)
    return orig_name, mask_name, overlay_name


def compute_slice_metrics(pred_map: np.ndarray, binary_mask: np.ndarray, threshold: float):
    """Compute morphometric metrics and confidence for a single slice."""
    tumor_pixels = int(np.sum(binary_mask > 0))
    total_pixels = int(binary_mask.shape[0] * binary_mask.shape[1])
    area_mm2 = round(tumor_pixels * PIXEL_AREA_MM2, 2)
    extruded_volume_cm3 = round((area_mm2 * SLICE_THICKNESS_MM) / 1000.0, 3)
    coverage_percent = round((tumor_pixels / float(total_pixels)) * 100.0, 4)

    if tumor_pixels > 0:
        ys, xs = np.where(binary_mask > 0)
        width_px = int(xs.max() - xs.min() + 1)
        height_px = int(ys.max() - ys.min() + 1)
        width_mm = round(width_px * PIXEL_SPACING_MM, 2)
        height_mm = round(height_px * PIXEL_SPACING_MM, 2)
        max_dimension_px = max(width_px, height_px)
        max_dimension_mm = round(max_dimension_px * PIXEL_SPACING_MM, 2)
        confidence = round(float(np.mean(pred_map[binary_mask > 0])), 4)
        mean_x = float(np.mean(xs))
        laterality = "Left Adrenal Gland" if mean_x >= (binary_mask.shape[1] / 2.0) else "Right Adrenal Gland"
    else:
        width_px = 0
        height_px = 0
        width_mm = 0.0
        height_mm = 0.0
        max_dimension_px = 0
        max_dimension_mm = 0.0
        laterality = "No Lesion Detected"
        raw_above = pred_map >= threshold
        if np.any(raw_above):
            confidence = round(min(float(np.mean(pred_map[raw_above])) * 0.5, threshold - 0.05), 4)
        else:
            confidence = round(float(np.percentile(pred_map, 99.5)), 4)

    return {
        "tumor_pixels": tumor_pixels,
        "area_mm2": area_mm2,
        "extruded_volume_cm3": extruded_volume_cm3,
        "width_px": width_px,
        "height_px": height_px,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "max_dimension_px": max_dimension_px,
        "max_dimension_mm": max_dimension_mm,
        "coverage_percent": coverage_percent,
        "confidence": confidence,
        "laterality": laterality,
    }


# =============================================================================
# PAGE ROUTES (Part 1, Part 2, Part 3)
# =============================================================================

@app.route("/", methods=["GET"])
def landing_page():
    return render_template("landing.html", user=get_current_user())


@app.route("/register", methods=["GET"])
def register_page():
    return render_template("register.html")


@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")


@app.route("/dashboard", methods=["GET"])
@login_required
def dashboard_page():
    return render_template("dashboard.html", user=get_current_user())


@app.route("/profile", methods=["GET"])
@login_required
def profile_page():
    return render_template("profile.html", user=get_current_user())


@app.route("/settings", methods=["GET"])
@login_required
def settings_page():
    return render_template("settings.html", user=get_current_user())


@app.route("/logout", methods=["GET"])
def logout_page():
    session.clear()
    return redirect(url_for("landing_page"))


# =============================================================================
# AUTHENTICATION API ENDPOINTS
# =============================================================================

@app.route("/api/profile", methods=["POST"])
@login_required
def api_update_profile():
    data = request.get_json(silent=True) or request.form
    full_name = (data.get("full_name") or "").strip()
    license_number = (data.get("license_number") or "").strip()
    institution = (data.get("institution") or "").strip()

    if len(full_name) < 3:
        return jsonify({"error": "Full Name must be at least 3 characters."}), 400
    if not license_number or not institution:
        return jsonify({"error": "License number and institution are required."}), 400

    conn = get_db_connection()
    with conn:
        conn.execute(
            "UPDATE users SET full_name = ?, license_number = ?, institution = ? WHERE id = ?",
            (full_name, license_number, institution, session["user_id"]),
        )
    conn.close()

    session["full_name"] = full_name
    session["license_number"] = license_number
    session["institution"] = institution
    return jsonify({"success": True, "user": get_current_user()})


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json(silent=True) or request.form

    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    license_number = (data.get("license_number") or "").strip()
    institution = (data.get("institution") or "").strip()
    password = data.get("password") or ""
    confirm_password = data.get("confirm_password") or ""

    if len(full_name) < 3:
        return jsonify({"error": "Full Name must be at least 3 characters."}), 400

    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        return jsonify({"error": "Please enter a valid email address."}), 400

    if not license_number:
        return jsonify({"error": "Medical License Number is required."}), 400

    if not institution:
        return jsonify({"error": "Hospital / Institution is required."}), 400

    if len(password) < 8 or not re.search(r"[A-Z]", password) or not re.search(r"\d", password):
        return jsonify({
            "error": "Password must be at least 8 characters and include at least 1 uppercase letter and 1 number."
        }), 400

    if password != confirm_password:
        return jsonify({"error": "Passwords do not match."}), 400

    password_hash = generate_password_hash(password)

    conn = get_db_connection()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO users (full_name, email, license_number, institution, password_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                (full_name, email, license_number, institution, password_hash),
            )
            user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({"error": "An account with this email already exists."}), 400
    conn.close()

    session.permanent = True
    session["user_id"] = user_id
    session["full_name"] = full_name
    session["email"] = email
    session["license_number"] = license_number
    session["institution"] = institution

    return jsonify({
        "success": True,
        "redirect": "/dashboard",
        "user": get_current_user(),
    })


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or request.form

    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    remember = bool(data.get("remember"))

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    conn = get_db_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()

    if row is None or not check_password_hash(row["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    session.permanent = True
    if remember:
        app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
    else:
        app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)

    session["user_id"] = row["id"]
    session["full_name"] = row["full_name"]
    session["email"] = row["email"]
    session["license_number"] = row["license_number"]
    session["institution"] = row["institution"]

    return jsonify({
        "success": True,
        "redirect": "/dashboard",
        "user": get_current_user(),
    })


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True, "redirect": "/"})


# =============================================================================
# CT SLICE ANALYSIS ENDPOINT (/analyze & /api/analyze) — 1 OR MORE SLICES
# =============================================================================

@app.route("/analyze", methods=["POST"])
@app.route("/api/analyze", methods=["POST"])
def analyze_bulk():
    # Parse threshold parameter
    try:
        threshold = float(request.form.get("threshold", PREDICTION_THRESHOLD))
        threshold = max(0.50, min(0.95, round(threshold, 2)))
    except (TypeError, ValueError):
        threshold = PREDICTION_THRESHOLD

    patient_input = {
        "id": (request.form.get("patient_id") or "").strip() or None,
        "age": (request.form.get("patient_age") or "").strip() or None,
        "sex": (request.form.get("patient_sex") or "").strip() or None,
    }

    # Check if re-analyzing the last uploaded slice(s) with a new threshold
    reanalyze_last = request.form.get("reanalyze") == "1"
    saved_slice_paths = []

    if reanalyze_last and os.path.isdir(LAST_BATCH_FOLDER):
        cached_files = sorted(
            [f for f in os.listdir(LAST_BATCH_FOLDER) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        )
        if len(cached_files) >= 1:
            saved_slice_paths = [
                (fname, os.path.join(LAST_BATCH_FOLDER, fname)) for fname in cached_files
            ]

    if not saved_slice_paths:
        # Collect uploaded files from 'ct_slices' or 'files' or 'file'
        files = request.files.getlist("ct_slices")
        if not files or all((not f or not f.filename) for f in files):
            files = request.files.getlist("files")
        if not files or all((not f or not f.filename) for f in files):
            files = request.files.getlist("file")

        files = [f for f in files if f and f.filename]

        if len(files) == 0:
            return jsonify({"error": "Please upload at least one slice"}), 400

        if len(files) > 1000:
            return jsonify({"error": "Maximum 1000 slices per analysis"}), 400

        # Sort uploaded files by filename
        def _clean_fname(fobj):
            raw = (fobj.filename or "").replace("\\", "/").split("/")[-1]
            return secure_filename(raw) or "slice.png"

        files = sorted(files, key=_clean_fname)

        # LAYER 1 Validation on all files first (catches wrong file formats like PDF immediately)
        for file in files:
            ok, msg = validate_file_properties(file)
            if not ok:
                return jsonify({"error": msg}), 400

        # Save files temporarily for Layer 2 & Layer 3 validation and processing
        batch_temp_dir = os.path.join(TEMP_FOLDER, f"upload_{int(time.time() * 1000)}")
        os.makedirs(batch_temp_dir, exist_ok=True)

        temp_saved = []
        try:
            for idx, file in enumerate(files):
                clean_name = _clean_fname(file)
                temp_path = os.path.join(batch_temp_dir, f"{idx:03d}_{clean_name}")
                file.save(temp_path)
                temp_saved.append((clean_name, temp_path))

            # VALIDATION Layers 2 & 3 on up to first 5 slices
            for clean_name, temp_path in temp_saved[:5]:
                # Layer 2: Dimensions, Grayscale check, Histogram check
                ok, msg = validate_image_characteristics(temp_path)
                if not ok:
                    shutil.rmtree(batch_temp_dir, ignore_errors=True)
                    return jsonify({"error": msg}), 400

                # Layer 3: Body-region classifier (adrenal CT vs chest X-ray / vehicle / non-CT)
                ok, msg = classify_body_region(temp_path)
                if not ok:
                    shutil.rmtree(batch_temp_dir, ignore_errors=True)
                    return jsonify({
                        "error": "Invalid image content",
                        "detail": f"{msg}. Uploaded images do not appear to be adrenal CT slices. Please upload a valid CT series.",
                    }), 400

            # Persist valid batch into LAST_BATCH_FOLDER so threshold slider re-runs work seamlessly
            shutil.rmtree(LAST_BATCH_FOLDER, ignore_errors=True)
            os.makedirs(LAST_BATCH_FOLDER, exist_ok=True)
            for idx, (clean_name, temp_path) in enumerate(temp_saved):
                persisted_path = os.path.join(LAST_BATCH_FOLDER, f"{idx:03d}_{clean_name}")
                shutil.copy2(temp_path, persisted_path)
                saved_slice_paths.append((clean_name, persisted_path))
        finally:
            shutil.rmtree(batch_temp_dir, ignore_errors=True)

    # Run U-Net segmentation across all slices in the series
    ensure_backend_ready()

    batch_out_id = f"series_{int(time.time() * 1000)}"
    batch_out_dir = os.path.join(STATIC_MASKS_FOLDER, batch_out_id)
    os.makedirs(batch_out_dir, exist_ok=True)

    results = []
    for i, (clean_name, slice_path) in enumerate(saved_slice_paths):
        file_hash = _compute_file_hash(slice_path)
        if file_hash in _prob_cache:
            pred_map, resized_gray = _prob_cache[file_hash]
        else:
            img_tensor, resized_gray = preprocess_slice(slice_path)
            pred_map = _predict_fn(img_tensor).numpy()[0, :, :, 0]
            _prob_cache[file_hash] = (pred_map, resized_gray)

        binary_mask = postprocess_mask(pred_map, threshold=threshold)
        metrics = compute_slice_metrics(pred_map, binary_mask, threshold=threshold)
        orig_name, mask_name, overlay_name = save_slice_visuals(resized_gray, binary_mask, batch_out_dir, i)

        results.append({
            "slice_index": i,
            "filename": clean_name,
            "area_mm2": metrics["area_mm2"],
            "tumor_pixels": metrics["tumor_pixels"],
            "extruded_volume_cm3": metrics["extruded_volume_cm3"],
            "width_mm": metrics["width_mm"],
            "width_px": metrics["width_px"],
            "height_mm": metrics["height_mm"],
            "height_px": metrics["height_px"],
            "max_dimension_mm": metrics["max_dimension_mm"],
            "max_dimension_px": metrics["max_dimension_px"],
            "coverage_percent": metrics["coverage_percent"],
            "confidence": metrics["confidence"],
            "laterality": metrics["laterality"],
            "original_url": f"/results/{batch_out_id}/{orig_name}",
            "mask_url": f"/results/{batch_out_id}/{mask_name}",
            "overlay_url": f"/results/{batch_out_id}/{overlay_name}",
        })

    # Compute series-level peak and total volume
    peak = max(results, key=lambda r: r["area_mm2"])
    total_volume_cm3 = round(
        sum(r["area_mm2"] for r in results) * SLICE_THICKNESS_MM / 1000.0, 3
    )

    is_positive = peak["area_mm2"] > 0.0
    detection_status = "positive" if is_positive else "negative"
    if is_positive:
        detection_confidence = round(float(peak["confidence"]), 3)
        tumor_location = peak["laterality"]
        estimated_size_mm = peak["max_dimension_mm"]
    else:
        detection_confidence = round(max(float(r["confidence"]) for r in results), 3)
        tumor_location = "No Lesion Detected"
        estimated_size_mm = 0.0

    return jsonify({
        "total_slices": len(results),
        "peak_slice_index": peak["slice_index"],
        "peak_area_mm2": peak["area_mm2"],
        "total_volume_cm3": total_volume_cm3,
        "detection_confidence": detection_confidence,
        "detection_status": detection_status,
        "threshold": threshold,
        "tumor_location": tumor_location,
        "estimated_size_mm": estimated_size_mm,
        "current_slice_index": peak["slice_index"],
        "patient_input": patient_input,
        "measurements": {
            "area_mm2": peak["area_mm2"],
            "area_px": peak["tumor_pixels"],
            "extruded_volume_cm3": peak["extruded_volume_cm3"],
            "width_mm": peak["width_mm"],
            "width_px": peak["width_px"],
            "height_mm": peak["height_mm"],
            "height_px": peak["height_px"],
            "max_dimension_mm": peak["max_dimension_mm"],
            "max_dimension_px": peak["max_dimension_px"],
            "coverage_percent": peak["coverage_percent"],
        },
        "slices": results,
    })


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/results/<path:filename>")
def result_file(filename):
    return send_from_directory(STATIC_MASKS_FOLDER, filename)


if __name__ == "__main__":
    print("\n" + "=" * 62)
    print("  AdrenalCare Server Ready -> http://127.0.0.1:5000")
    print("=" * 62 + "\n")
    threading.Thread(target=ensure_backend_ready, daemon=True).start()
    app.run(debug=True, use_reloader=False)