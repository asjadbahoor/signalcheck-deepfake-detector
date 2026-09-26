"""
Deepfake Detector -- Streamlit app.

Run with:
    streamlit run app.py

See README.md for setup, training data sources, and how the continuous
learning / feedback loop works.
"""
import streamlit as st
from PIL import Image

import config
import feedback_manager
from model import DeepfakeDetector

st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="centered")


@st.cache_resource
def load_detector():
    return DeepfakeDetector()


detector = load_detector()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🕵️ Deepfake & Fraud Image Detector")
st.caption(
    "Upload an image to check whether it looks real or AI-manipulated, "
    "with a confidence score. Built to help flag misinformation and "
    "fraudulent media before it spreads."
)

if not st.session_state.get("_warned_no_checkpoint"):
    import os
    if not os.path.exists(config.CURRENT_MODEL_PATH):
        st.warning(
            "⚠️ No trained checkpoint found yet. This demo is currently running on a "
            "generic ImageNet backbone that has **not** been trained to tell real "
            "faces from deepfakes -- predictions below are not meaningful until you "
            "run `python train.py` on a labeled dataset. See README.md.",
            icon="⚠️",
        )
    st.session_state["_warned_no_checkpoint"] = True

# ---------------------------------------------------------------------------
# Sidebar: model & feedback status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📊 Model status")
    stats = feedback_manager.get_stats()

    st.metric("Feedback samples collected", stats["total_feedback"])
    st.metric("Corrections received", stats["corrections_received"])

    progress = min(stats["pending_feedback"] / config.RETRAIN_THRESHOLD, 1.0)
    st.progress(progress, text=(
        f"{stats['pending_feedback']} / {config.RETRAIN_THRESHOLD} new samples "
        f"toward next retrain"
    ))

    if stats["last_retrain"]:
        ts, accepted, new_acc = stats["last_retrain"]
        status = "✅ accepted" if accepted else "❌ rejected (safety check failed)"
        acc_str = f"{new_acc:.1%}" if new_acc is not None else "n/a"
        st.caption(f"Last retrain: {ts[:19]} UTC -- {status} (val acc: {acc_str})")
    else:
        st.caption("No retraining has happened yet.")

    st.divider()
    if st.button("🔁 Retrain now (force)", use_container_width=True,
                  help="Runs fine-tuning immediately on all pending feedback, "
                       "even if the threshold hasn't been reached."):
        with st.spinner("Fine-tuning on pending feedback and validating..."):
            result = feedback_manager.run_retrain(force=True)
        if result.get("skipped"):
            st.info(result.get("reason", "Nothing to retrain on yet."))
        elif result.get("accepted"):
            detector.reload()
            st.success(f"Model updated ✅ ({result['reason']})")
        else:
            st.error(f"Update rejected: {result.get('reason')}")

    st.divider()
    st.caption(
        "**How learning works:** every confirmed/corrected prediction is queued. "
        "Once enough pile up, the model is fine-tuned on a *copy* and only "
        "deployed if it doesn't lose accuracy on a held-out validation set -- "
        "this stops a single mistaken or malicious label from corrupting the model."
    )

# ---------------------------------------------------------------------------
# Main: upload + predict
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        result = detector.predict(image)

    label = result["label"]
    confidence = result["confidence"]

    if label == "fake":
        st.error(f"### 🚨 Likely DEEPFAKE / manipulated — {confidence:.1%} confidence")
        if confidence >= 0.85:
            st.markdown(
                "⚠️ **High-confidence deepfake detection.** If this was shared as "
                "genuine (e.g. as news, a testimonial, or proof of identity), treat "
                "it as a strong signal of misinformation or fraud -- verify through "
                "an independent source before acting on it."
            )
    else:
        st.success(f"### ✅ Likely REAL — {confidence:.1%} confidence")

    if confidence < config.LOW_CONFIDENCE_THRESHOLD:
        st.info(
            "🤔 This prediction has relatively low confidence. Your feedback below "
            "is especially valuable for improving the model on cases like this."
        )

    st.progress(confidence, text=f"Confidence: {confidence:.1%}")
    with st.expander("See full probability breakdown"):
        for cls_name, prob in result["probabilities"].items():
            st.write(f"**{cls_name.capitalize()}**: {prob:.1%}")

    st.divider()
    st.subheader("Was this correct?")
    st.caption(
        "Your answer is stored to help retrain the model (see sidebar). "
        "Uploaded images used for feedback are kept locally in `data/feedback/`."
    )

    col1, col2 = st.columns(2)
    feedback_given_key = f"feedback_given_{uploaded_file.file_id}"

    if not st.session_state.get(feedback_given_key):
        with col1:
            if st.button("✅ Yes, correct", use_container_width=True, key="correct_btn"):
                feedback_manager.add_feedback(image, label, confidence, correct_label=label)
                st.session_state[feedback_given_key] = True
                st.rerun()
        with col2:
            if st.button("❌ No, incorrect", use_container_width=True, key="incorrect_btn"):
                st.session_state[f"show_correction_{uploaded_file.file_id}"] = True

        if st.session_state.get(f"show_correction_{uploaded_file.file_id}"):
            correct_label = st.radio(
                "What's the correct label?", options=config.CLASS_NAMES,
                horizontal=True, key="correction_radio",
            )
            if st.button("Submit correction", key="submit_correction_btn"):
                feedback_manager.add_feedback(image, label, confidence, correct_label=correct_label)
                st.session_state[feedback_given_key] = True
                st.rerun()
    else:
        st.success("Thanks -- feedback recorded.")
        if feedback_manager.ready_to_retrain():
            with st.spinner("Enough feedback collected -- fine-tuning and validating a model update..."):
                retrain_result = feedback_manager.run_retrain(force=False)
            if retrain_result.get("accepted"):
                detector.reload()
                st.toast("Model updated with new feedback! 🎉", icon="🎉")
            elif not retrain_result.get("skipped"):
                st.toast(f"Retrain attempted but rejected: {retrain_result.get('reason')}", icon="⚠️")

# ---------------------------------------------------------------------------
# Footer / responsible use notice
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ Limitations & responsible use"):
    st.markdown(
        "- This tool gives a **statistical estimate**, not a certified forensic "
        "verdict. Treat results as one signal among several, especially for "
        "high-stakes decisions (fraud claims, content moderation, legal use).\n"
        "- Accuracy depends entirely on the quality and diversity of the training "
        "data. A model trained on one deepfake generation method may not catch "
        "images from newer or different methods.\n"
        "- False positives (real images flagged as fake) and false negatives "
        "(fake images flagged as real) are both possible -- always allow for human review.\n"
        "- Images submitted as feedback are stored locally to improve the model. "
        "Don't upload images you don't have the right to use for this purpose."
    )
