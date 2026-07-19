"""
Handwritten Digit Recognition - Streamlit App
Loads the trained ANN model (ann_model.keras) and predicts digits from:
  1. A hand-drawn canvas (draw with mouse/touch)
  2. An uploaded image file

How to run:
    pip install streamlit tensorflow numpy pillow streamlit-drawable-canvas --break-system-packages
    streamlit run app.py
"""

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ----------------------------------------------------------------------
# Page Config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Digit Recognition", page_icon="✏️", layout="centered")

# ----------------------------------------------------------------------
# Load Model (cached so it only loads once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("ann_model.keras")

model = load_model()

# ----------------------------------------------------------------------
# Preprocessing (must match training: normalize 0-1, shape 28x28)
# ----------------------------------------------------------------------
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("L")                  # grayscale
    img = img.resize((28, 28))              # match MNIST size
    arr = np.array(img).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28)            # ANN expects (28,28), no channel dim
    return arr

def predict(arr: np.ndarray):
    probs = model.predict(arr, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs)) * 100
    return digit, confidence, probs

# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("✏️ Handwritten Digit Recognition")
st.caption("ANN model trained on MNIST — 97.84% test accuracy")

tab1, tab2 = st.tabs(["🖌️ Draw a Digit", "📁 Upload an Image"])

# ---- Tab 1: Draw ----
with tab1:
    st.write("Draw a single digit (0-9) below, then click **Predict**.")

    col1, col2 = st.columns([2, 1])
    with col1:
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=18,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )

    with col2:
        st.write("")
        predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)
        clear_note = st.caption("Use the canvas toolbar (top-right of canvas) to clear/undo.")

    if predict_btn:
        if canvas_result.image_data is not None:
            img = Image.fromarray(canvas_result.image_data.astype("uint8"), mode="RGBA")
            arr = preprocess(img)
            digit, confidence, probs = predict(arr)

            st.success(f"### Predicted Digit: {digit}")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.bar_chart({"Probability": probs}, use_container_width=True)
        else:
            st.warning("Please draw a digit first.")

# ---- Tab 2: Upload ----
with tab2:
    st.write("Upload an image of a handwritten digit (black digit on white background, or vice versa).")
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    invert = st.checkbox("Invert colors (use if your digit is black on white background)", value=True)

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", width=150)

        if invert:
            img_for_model = ImageOps.invert(img.convert("L"))
        else:
            img_for_model = img

        if st.button("🔍 Predict Digit", type="primary"):
            arr = preprocess(img_for_model)
            digit, confidence, probs = predict(arr)

            st.success(f"### Predicted Digit: {digit}")
            st.write(f"Confidence: **{confidence:.2f}%**")
            st.bar_chart({"Probability": probs}, use_container_width=True)

# ----------------------------------------------------------------------
# Sidebar - Model Info
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Model Info")
    st.write("**Architecture:** ANN (Artificial Neural Network)")
    st.markdown(
        "- Flatten (28×28 → 784)\n"
        "- Dense(128, ReLU)\n"
        "- Dense(64, ReLU)\n"
        "- Dense(10, Softmax)"
    )
    st.write("**Dataset:** MNIST (60,000 train / 10,000 test)")
    st.write("**Test Accuracy:** 97.84%")
    st.write("**Optimizer:** Adam")
    st.write("**Loss:** Sparse Categorical Crossentropy")
