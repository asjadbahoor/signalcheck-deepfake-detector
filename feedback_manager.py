"""
Feedback loop plumbing.

Every prediction the app makes can be confirmed or corrected by the user.
That verdict (always supplied by a human, never inferred automatically) is
stored here. Once enough new confirmed samples pile up, the app can trigger
train.fine_tune(), which validates the result before it's ever deployed
(see train.py for the accuracy safety check).

This module deliberately does NOT retrain on every single click -- see
config.RETRAIN_THRESHOLD -- both to keep retraining meaningful and to avoid
a single user (malicious or just wrong) swinging the model on their own.
"""
import os
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone

from PIL import Image

import config
import train as train_module


def _get_conn():
    os.makedirs(os.path.dirname(config.DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            correct_label TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            used_in_training INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS retrain_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            num_samples INTEGER NOT NULL,
            old_val_acc REAL,
            new_val_acc REAL,
            accepted INTEGER NOT NULL,
            reason TEXT
        )
    """)
    return conn


def add_feedback(image: Image.Image, predicted_label: str, confidence: float, correct_label: str) -> None:
    """Records one human-confirmed (or human-corrected) prediction."""
    os.makedirs(config.FEEDBACK_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(config.FEEDBACK_DIR, filename)
    image.convert("RGB").save(save_path, "JPEG", quality=95)

    conn = _get_conn()
    with conn:
        conn.execute(
            "INSERT INTO feedback (image_path, predicted_label, confidence, correct_label, timestamp, used_in_training) "
            "VALUES (?, ?, ?, ?, ?, 0)",
            (save_path, predicted_label, confidence, correct_label,
             datetime.now(timezone.utc).isoformat()),
        )
    conn.close()


def get_stats() -> dict:
    """Returns counts used to drive the sidebar progress bar in the app."""
    conn = _get_conn()
    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    pending = conn.execute("SELECT COUNT(*) FROM feedback WHERE used_in_training = 0").fetchone()[0]
    corrections = conn.execute(
        "SELECT COUNT(*) FROM feedback WHERE predicted_label != correct_label"
    ).fetchone()[0]
    last_retrain = conn.execute(
        "SELECT timestamp, accepted, new_val_acc FROM retrain_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return {
        "total_feedback": total,
        "pending_feedback": pending,
        "corrections_received": corrections,
        "retrain_threshold": config.RETRAIN_THRESHOLD,
        "last_retrain": last_retrain,  # (timestamp, accepted, new_val_acc) or None
    }


def ready_to_retrain() -> bool:
    return get_stats()["pending_feedback"] >= config.RETRAIN_THRESHOLD


def _stage_pending_feedback(conn) -> tuple[str, list[int]]:
    """Copies not-yet-used feedback images into a temp real/fake folder
    structure that train.fine_tune() (via ImageFolder) can consume."""
    rows = conn.execute(
        "SELECT id, image_path, correct_label FROM feedback WHERE used_in_training = 0"
    ).fetchall()

    staging_dir = os.path.join(config.FEEDBACK_DIR, "_staging")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    for cls in config.CLASS_NAMES:
        os.makedirs(os.path.join(staging_dir, cls), exist_ok=True)

    row_ids = []
    for row_id, image_path, correct_label in rows:
        if not os.path.exists(image_path):
            continue
        dest = os.path.join(staging_dir, correct_label, os.path.basename(image_path))
        shutil.copy2(image_path, dest)
        row_ids.append(row_id)

    return staging_dir, row_ids


def run_retrain(force: bool = False) -> dict:
    """Attempts a fine-tuning pass on all pending feedback.

    If `force` is False, does nothing (and returns {"skipped": True}) unless
    the pending count has reached config.RETRAIN_THRESHOLD -- this is what
    the automatic "learns as it's used" behavior calls under the hood.
    """
    stats = get_stats()
    if not force and stats["pending_feedback"] < config.RETRAIN_THRESHOLD:
        return {"skipped": True, "pending_feedback": stats["pending_feedback"],
                "threshold": config.RETRAIN_THRESHOLD}

    conn = _get_conn()
    staging_dir, row_ids = _stage_pending_feedback(conn)

    if not row_ids:
        conn.close()
        return {"skipped": True, "reason": "No usable pending feedback images found on disk."}

    result = train_module.fine_tune(staging_dir)

    with conn:
        conn.execute(
            "INSERT INTO retrain_log (timestamp, num_samples, old_val_acc, new_val_acc, accepted, reason) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), len(row_ids),
             result.get("old_val_acc"), result.get("new_val_acc"),
             int(result.get("accepted", False)), result.get("reason", "")),
        )
        # Mark these rows as used regardless of accept/reject so a rejected
        # batch doesn't get retried forever -- it's preserved in retrain_log
        # for auditing either way.
        conn.executemany(
            "UPDATE feedback SET used_in_training = 1 WHERE id = ?",
            [(rid,) for rid in row_ids],
        )
    conn.close()

    shutil.rmtree(staging_dir, ignore_errors=True)
    result["num_samples"] = len(row_ids)
    return result
