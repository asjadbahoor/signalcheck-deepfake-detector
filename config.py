"""
Central configuration for the Deepfake Detector app.
Edit values here rather than hunting through the codebase.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")       # data/train/real, data/train/fake
VAL_DIR = os.path.join(DATA_DIR, "val")           # data/val/real,   data/val/fake
FEEDBACK_DIR = os.path.join(DATA_DIR, "feedback") # images submitted by users, sorted after review

MODELS_DIR = os.path.join(BASE_DIR, "models")
CURRENT_MODEL_PATH = os.path.join(MODELS_DIR, "current_model.pt")
BACKUP_MODEL_PATH = os.path.join(MODELS_DIR, "previous_model.pt")

DB_PATH = os.path.join(BASE_DIR, "feedback.db")

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
# Folder names expected under data/train and data/val. The actual
# index<->label mapping used at inference time is read from each checkpoint
# (see model.save_checkpoint / DeepfakeDetector), not hardcoded here, so
# training is safe regardless of how ImageFolder happens to sort classes.
CLASS_NAMES = ["real", "fake"]
BACKBONE = "efficientnet_b0"            # torchvision model name

INITIAL_TRAIN_EPOCHS = 10
INITIAL_TRAIN_LR = 1e-4
INITIAL_BATCH_SIZE = 32

# Fine-tuning (continuous learning) is deliberately gentler than initial
# training: smaller learning rate, fewer epochs, so a handful of new
# feedback samples can't wildly swing the model.
FINE_TUNE_EPOCHS = 3
FINE_TUNE_LR = 1e-5
FINE_TUNE_BATCH_SIZE = 16

# ---------------------------------------------------------------------------
# Continuous learning / feedback loop settings
# ---------------------------------------------------------------------------
# Number of NEW human-confirmed feedback samples required before the app
# offers to retrain. Keeps retraining meaningful and prevents thrashing
# the model after every single click.
RETRAIN_THRESHOLD = 25

# After fine-tuning, the candidate model must score at least this close to
# (or better than) the previous model's validation accuracy, or the update
# is rejected and the previous model is kept. This is the main guardrail
# against feedback poisoning or accidental degradation.
MAX_ALLOWED_ACCURACY_DROP = 0.02  # 2 percentage points

# A prediction below this confidence is flagged in the UI as "uncertain"
# and users are especially encouraged to give feedback on it, since these
# are the most valuable samples for retraining.
LOW_CONFIDENCE_THRESHOLD = 0.65
