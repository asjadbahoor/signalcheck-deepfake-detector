"""
Deepfake Detector -- Flask + HTML/CSS/JS frontend.

This is an alternative to app.py (the Streamlit version) with a fully
custom, polished interface. It uses the exact same model, training and
feedback-loop code (model.py, train.py, feedback_manager.py, config.py) --
only the presentation layer differs.

Run with:
    python app_flask.py
then open http://127.0.0.1:5000
"""
import os
import time
import uuid

from flask import Flask, jsonify, render_template, request
from PIL import Image

import config
import feedback_manager
from model import DeepfakeDetector

app = Flask(__name__)
detector = DeepfakeDetector()

TMP_DIR = os.path.join(config.DATA_DIR, "_tmp_uploads")
os.makedirs(TMP_DIR, exist_ok=True)
TMP_MAX_AGE_SECONDS = 2 * 60 * 60  # 2 hours


def _cleanup_tmp():
    """Best-effort removal of stale temp uploads (e.g. abandoned sessions)."""
    now = time.time()
    try:
        for name in os.listdir(TMP_DIR):
            path = os.path.join(TMP_DIR, name)
            if os.path.isfile(path) and now - os.path.getmtime(path) > TMP_MAX_AGE_SECONDS:
                os.remove(path)
    except OSError:
        pass


@app.route("/")
def index():
    return render_template("index.html", retrain_threshold=config.RETRAIN_THRESHOLD)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    _cleanup_tmp()

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded."}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream)
        image.load()
    except Exception:
        return jsonify({"error": "That file couldn't be read as an image."}), 400

    result = detector.predict(image)

    # Stash the image on disk under a token so /api/feedback can retrieve the
    # exact same bytes later without the browser having to re-upload it.
    token = uuid.uuid4().hex
    tmp_path = os.path.join(TMP_DIR, f"{token}.jpg")
    image.convert("RGB").save(tmp_path, "JPEG", quality=95)

    result["token"] = token
    result["low_confidence"] = result["confidence"] < config.LOW_CONFIDENCE_THRESHOLD
    return jsonify(result)


@app.route("/api/feedback", methods=["POST"])
def api_feedback():
    payload = request.get_json(silent=True) or {}
    token = payload.get("token")
    predicted_label = payload.get("predicted_label")
    confidence = payload.get("confidence")
    correct_label = payload.get("correct_label")

    if not all([token, predicted_label, correct_label]) or confidence is None:
        return jsonify({"error": "Missing fields."}), 400
    if correct_label not in config.CLASS_NAMES:
        return jsonify({"error": "Invalid label."}), 400

    tmp_path = os.path.join(TMP_DIR, f"{token}.jpg")
    if not os.path.exists(tmp_path):
        return jsonify({"error": "This image has expired -- please analyze it again."}), 410

    image = Image.open(tmp_path)
    feedback_manager.add_feedback(image, predicted_label, float(confidence), correct_label)
    os.remove(tmp_path)

    retrain_result = None
    if feedback_manager.ready_to_retrain():
        retrain_result = feedback_manager.run_retrain(force=False)
        if retrain_result.get("accepted"):
            detector.reload()

    return jsonify({"ok": True, "retrain": retrain_result})


@app.route("/api/stats")
def api_stats():
    return jsonify(feedback_manager.get_stats())


@app.route("/api/retrain", methods=["POST"])
def api_retrain():
    result = feedback_manager.run_retrain(force=True)
    if result.get("accepted"):
        detector.reload()
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=5000)
