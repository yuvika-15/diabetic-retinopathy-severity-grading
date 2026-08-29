"""
Diabetic Retinopathy Severity Grading — Deployment Demo

Upload a retinal fundus photo, get a predicted severity grade (0-4),
a Grad-CAM overlay showing model attention, and a referral flag.

Run locally:   streamlit run app.py
Deploy free:   push this repo to GitHub, then deploy on
               streamlit.io/cloud or huggingface.co/spaces (Streamlit SDK)
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

sys.path.append("src/")
from preprocessing import full_pipeline_preprocess  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH = "outputs/models/phase2_finetuned_oversampled.keras"
# Oversampled model chosen as primary: higher QWK (0.854 vs 0.826) and better
# Grade 4 recall (0.424 vs 0.333) than the class-weighted baseline. See
# README's "Class Imbalance Experiment" section for the full comparison.

GRADE_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}
REFERRAL_GRADES = {3, 4}  # matches the project's triage framing


# ---------------------------------------------------------------------------
# Model loading (cached so it only happens once per session, not per upload)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_backbone():
    model = tf.keras.models.load_model(MODEL_PATH)
    # Pull the nested EfficientNetB0 submodel out of the loaded model for
    # Grad-CAM — see 03_model_training.ipynb for why this must be extracted
    # from the loaded model rather than re-instantiated separately.
    backbone_candidates = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)]
    base_model = backbone_candidates[0]
    return model, base_model


def get_head_layers(model):
    gap_layer, dense_layer = None, None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            gap_layer = layer
        elif isinstance(layer, tf.keras.layers.Dense):
            dense_layer = layer
    return gap_layer, dense_layer


# ---------------------------------------------------------------------------
# Grad-CAM (identical logic to the training notebook)
# ---------------------------------------------------------------------------

def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name="top_activation"):
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    conv_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output],
    )

    gap_layer, dense_layer = get_head_layers(model)

    with tf.GradientTape() as tape:
        conv_outputs, backbone_output = conv_model(img_array, training=False)
        x = gap_layer(backbone_output)
        predictions = dense_layer(x)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    conv_outputs = tf.cast(conv_outputs[0], tf.float32)
    grads = tf.cast(grads, tf.float32)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), predictions.numpy()[0], int(pred_index)


def overlay_heatmap(img_uint8, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)


def analyze_attention_localization(heatmap, threshold=0.5):
    """
    Rough proxy for how concentrated vs. diffuse the Grad-CAM heatmap is:
    the fraction of the heatmap above `threshold` after normalization.
    Motivated by the project's own finding that near-miss errors showed
    attention aligned with real lesions, while the most severe errors
    showed diffuse, unfocused attention (see README's Grad-CAM section).
    This is a heuristic, not a validated clinical signal.
    """
    active_fraction = float(np.mean(heatmap > threshold))
    if active_fraction < 0.15:
        label = "highly localized"
    elif active_fraction < 0.35:
        label = "moderately localized"
    else:
        label = "diffuse / spread out"
    return label, active_fraction


def generate_conclusion(pred_class, probs, heatmap):
    confidence = float(np.max(probs))
    grade_name = GRADE_NAMES[pred_class]
    referral = pred_class in REFERRAL_GRADES

    if confidence >= 0.75:
        conf_level = "high"
    elif confidence >= 0.5:
        conf_level = "moderate"
    else:
        conf_level = "low"

    localization, active_fraction = analyze_attention_localization(heatmap)

    lines = [
        f"The model predicts **Grade {pred_class} — {grade_name}** with "
        f"**{conf_level} confidence** ({confidence:.0%})."
    ]

    if referral:
        lines.append(
            "This falls in the **referral range** (Severe / Proliferative DR) — "
            "the triage logic flags this case for specialist review."
        )
    else:
        lines.append(
            "This falls in the **non-referral range** based on this prediction alone."
        )

    lines.append(
        f"Grad-CAM attention is **{localization}** "
        f"(highlighted region covers ~{active_fraction:.0%} of the retina)."
    )

    if localization == "diffuse / spread out":
        lines.append(
            "Diffuse attention was associated with the least reliable predictions "
            "in this project's evaluation — treat this result with extra caution."
        )

    if pred_class == 0 and conf_level == "high":
        lines.append(
            "**Caution:** this model has a documented failure mode where it can "
            "confidently predict 'No DR' while anchoring on generic retinal "
            "landmarks (e.g. the optic disc) rather than genuine absence of "
            "disease — observed on a small number of severe cases during "
            "evaluation. A confident 'No DR' result should not be treated as "
            "fully conclusive without clinical correlation."
        )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="DR Severity Grading", layout="wide")

st.title("Diabetic Retinopathy Severity Grading")
st.caption(
    "Portfolio / research demo. **Not a medical device — do not use for real "
    "diagnosis or clinical decisions.**"
)

with st.sidebar:
    st.header("About")
    st.write(
        "EfficientNetB0 transfer-learning model trained on the APTOS 2019 "
        "dataset, graded 0 (No DR) to 4 (Proliferative DR). Framed as a "
        "referral-triage tool: Grades 3-4 are flagged for specialist review."
    )
    st.markdown("**Known limitation:** Grad-CAM analysis found this model "
                "can occasionally anchor on generic retinal landmarks (e.g. "
                "the optic disc) rather than disease-specific features, "
                "producing confident but wrong 'No DR' predictions on a "
                "small number of severe cases. See the project README for "
                "the full interpretability writeup.")
    st.markdown("[View project on GitHub](#)")  # replace with your repo link

if not os.path.exists(MODEL_PATH):
    st.error(f"Model checkpoint not found at `{MODEL_PATH}`. "
              "Run the training notebooks first, or update MODEL_PATH.")
    st.stop()

model, base_model = load_model_and_backbone()

uploaded_file = st.file_uploader("Upload a retinal fundus photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Could not read this file as an image. Please upload a valid PNG/JPG.")
        st.stop()

    with st.spinner("Processing..."):
        processed = full_pipeline_preprocess(img_bgr)
        img_input = np.expand_dims(processed, axis=0)
        heatmap, probs, pred_class = make_gradcam_heatmap(img_input, model, base_model)

    display_img = np.clip(processed, 0, 255).astype(np.uint8)
    overlay = overlay_heatmap(display_img, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(display_img, caption="Preprocessed image", use_container_width=True)
    with col2:
        st.image(overlay, caption="Grad-CAM — model attention", use_container_width=True)

    st.subheader(f"Predicted grade: {pred_class} — {GRADE_NAMES[pred_class]}")

    if pred_class in REFERRAL_GRADES:
        st.error("Referral recommended — flagged as Severe / Proliferative DR")
    else:
        st.success("No urgent referral flagged based on this prediction")

    st.write("Class probabilities:")
    prob_dict = {f"G{i}: {GRADE_NAMES[i]}": float(probs[i]) for i in range(5)}
    st.bar_chart(prob_dict)

    st.subheader("Interpretation")
    st.markdown(generate_conclusion(pred_class, probs, heatmap))