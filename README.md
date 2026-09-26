# Deepfake & Fraud Image Detector

An interactive Streamlit app that classifies an uploaded image as **real** or
**likely deepfake/manipulated**, with a confidence score -- built to help
flag misinformation and fraudulent media on social platforms before it
spreads. The model is designed to keep improving from real usage through a
supervised feedback loop (details below).

> ⚠️ **Important:** this repo ships the app, the model architecture, and the
> full training/feedback pipeline -- but **not a trained model**. Out of the
> box it runs on a generic ImageNet backbone that has not learned to tell
> real faces from deepfakes. You need to train it on a real labeled dataset
> first (see below) before predictions mean anything.

## How it works

- **Model**: EfficientNet-B0, pretrained on ImageNet, fine-tuned with a small
  classifier head for binary real/fake classification. Transfer learning is
  used because training a strong image classifier from scratch needs far
  more data/compute than most people have for this task.
- **App**: Streamlit UI -- upload an image, get a label + confidence score,
  color-coded and with a plain-language warning for high-confidence
  deepfakes.
- **Continuous learning**: every prediction can be confirmed or corrected by
  the user. Confirmed feedback is queued, and once enough new samples pile
  up, the app fine-tunes a *copy* of the live model and only deploys it if
  it doesn't lose accuracy on a held-out validation set. This is what keeps
  a handful of wrong or malicious labels from corrupting the model -- see
  "Safety guardrails" below.

## Project structure

```
deepfake_detector/
├── app.py               # Streamlit UI
├── app_flask.py          # Flask API + custom HTML/CSS/JS UI (alternative frontend)
├── templates/index.html  # Flask frontend markup
├── static/css/style.css  # Flask frontend styling
├── static/js/app.js      # Flask frontend behavior
├── model.py             # Model architecture + inference wrapper
├── train.py             # Initial training + fine_tune() for the feedback loop
├── data_utils.py         # Dataset loading (ImageFolder-based)
├── feedback_manager.py  # SQLite feedback store + retrain trigger
├── config.py             # All paths & hyperparameters in one place
├── requirements.txt
├── data/
│   ├── train/real/  train/fake/    <- put your training images here
│   ├── val/real/    val/fake/      <- held-out set, used as the safety check
│   ├── feedback/                   <- images submitted via either app
│   └── _tmp_uploads/                <- Flask-only, short-lived staging (auto-cleaned)
└── models/
    └── current_model.pt            <- created after you run train.py
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

GPU is optional but strongly recommended for training (CUDA-enabled PyTorch
build). Inference in the app is fast enough on CPU.

## 1. Get a labeled dataset

You need images sorted into `data/train/real/`, `data/train/fake/`,
`data/val/real/`, `data/val/fake/` (val = a held-out slice, e.g. 10-15% of
your data, never touched by training). Some well-known public research
datasets for this task:

- **FaceForensics++** -- widely used academic benchmark with multiple
  manipulation methods (Deepfakes, Face2Face, FaceSwap, NeuralTextures).
- **DFDC (Deepfake Detection Challenge)** -- large-scale dataset released by
  Meta/AWS/Microsoft/academic partners for a Kaggle competition.
- **Celeb-DF** -- celebrity deepfake videos plus real counterparts.
- **140k Real and Fake Faces** (Kaggle) -- a simpler, already-split
  still-image dataset, good for a quick first pass.

Most of these are video datasets -- extract frames as JPG stills for this
image-based app. Follow each dataset's own license/access terms (several
require filling out an academic-use request form).

## 2. Train the initial model

```bash
python train.py --data_dir data --epochs 10
```

This trains from the ImageNet-pretrained backbone, prints train/val accuracy
per epoch, and saves `models/current_model.pt`. Adjust `--epochs`, `--lr`,
`--batch_size` as needed; defaults live in `config.py`.

## 3. Run the app

There are two interchangeable frontends. Both call the exact same model,
training, and feedback-loop code (`model.py`, `train.py`,
`feedback_manager.py`, `config.py`) -- only the presentation differs.

**Streamlit** -- fastest to run, good for quick testing/demos:

```bash
streamlit run app.py
```

**Flask + custom HTML/CSS/JS** -- a polished, fully custom interface
(drag-and-drop upload, animated scan indicator, a "model ledger" sidebar):

```bash
python app_flask.py
```

then open `http://127.0.0.1:5000`. Files live in `templates/index.html`,
`static/css/style.css`, and `static/js/app.js` if you want to restyle it or
plug it into a bigger site.

Either way: upload an image, get a real/fake verdict with a confidence bar,
and confirm or correct it -- that's what feeds the continuous learning loop.

## Continuous learning: safety guardrails

"Learning from usage" is powerful but risky if done naively -- a single bad
actor could otherwise feed it wrong labels and degrade it over time. This
implementation is deliberately conservative:

1. Every feedback entry is a **human-confirmed or human-corrected** label --
   nothing is inferred automatically.
2. New feedback is only used once **`config.RETRAIN_THRESHOLD`** (default
   25) new samples have accumulated, and fine-tuning uses a much smaller
   learning rate (`config.FINE_TUNE_LR`) and fewer epochs than initial
   training, so it nudges rather than overwrites the model.
3. Fine-tuning always happens on a **copy** of the current model. The
   candidate is evaluated on `data/val` (never touched by feedback), and is
   only deployed if its accuracy doesn't drop by more than
   `config.MAX_ALLOWED_ACCURACY_DROP` (default 2 points) versus the current
   model. Otherwise the update is rejected and logged.
4. Every retrain attempt -- accepted or rejected -- is recorded in the
   `retrain_log` table in `feedback.db` for auditing.
5. The previous model is always backed up to `models/previous_model.pt`
   before a successful update, so you can roll back manually if needed.

For a production deployment handling real user traffic, consider adding:
manual admin review of feedback before it's used, rate-limiting feedback per
user/IP, and periodic re-evaluation against a curated "golden set" of known
tricky examples.

## Limitations & responsible use

- This is a **statistical estimate**, not a certified forensic verdict.
  Treat results as one signal among several -- especially for high-stakes
  use (fraud claims, content moderation, legal proceedings).
- A model trained on one generation/manipulation method may not generalize
  to images produced by newer or different deepfake techniques. Retrain
  periodically as new manipulation methods emerge.
- Both false positives (flagging real images as fake) and false negatives
  are possible. Always keep a human in the loop for consequential decisions.
- Respect image rights and privacy: only use images you're allowed to
  process, and be transparent with end users that submitted images may be
  stored (see `data/feedback/`) to improve the model.

## Extension ideas

- Add Grad-CAM visualization so users can see *which regions* drove the
  prediction (useful for trust and for spotting model blind spots).
- Extend to video (frame sampling + temporal consistency checks) and audio
  deepfakes.
- Ensemble multiple architectures/checkpoints for a more robust confidence
  score.
- Add an admin review queue for feedback before it's eligible for
  retraining, instead of trusting every submitter.
