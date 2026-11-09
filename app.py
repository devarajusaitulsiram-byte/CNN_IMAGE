import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="CNN-Based Image Classification System", page_icon="🧠", layout="wide")
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
CNN_PATH = MODEL_DIR / "best_cnn_model.keras"
MOBILENET_PATH = MODEL_DIR / "best_mobilenet_model.keras"

CLASS_NAMES = ["Airplane", "Automobile", "Bird", "Cat", "Deer",
               "Dog", "Frog", "Horse", "Ship", "Truck"]

st.title("CNN-Based Image Classification System")
st.write("Upload an image to classify it into one of the 10 CIFAR-10 categories.")

@st.cache_resource(show_spinner="Loading trained models...")
def load_models():
    missing = [p.name for p in (CNN_PATH, MOBILENET_PATH) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + ", ".join(missing) +
            ". Upload both .keras files into the repository's models/ folder."
        )
    return (
        tf.keras.models.load_model(CNN_PATH, compile=False),
        tf.keras.models.load_model(MOBILENET_PATH, compile=False),
    )

try:
    cnn_model, mobilenet_model = load_models()
    st.success("Both trained models loaded successfully.")
except Exception as exc:
    st.error(f"Could not load the trained models: {exc}")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])
    model_choice = st.selectbox("Select classification model", ["MobileNetV2", "Custom CNN"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify image", type="primary"):
        try:
            if model_choice == "MobileNetV2":
                model = mobilenet_model
                resized = image.resize((96, 96))
                arr = np.asarray(resized, dtype=np.float32)
                arr = (arr / 127.5) - 1.0
            else:
                model = cnn_model
                resized = image.resize((32, 32))
                arr = np.asarray(resized, dtype=np.float32) / 255.0

            scores = np.asarray(model.predict(np.expand_dims(arr, axis=0), verbose=0))[0]
            if scores.shape[0] != 10:
                st.error(f"Expected 10 output scores, received {scores.shape[0]}.")
                st.stop()

            top = np.argsort(scores)[::-1][:3]
            with col2:
                st.subheader("Top 3 predictions")
                best = int(top[0])
                st.metric("Predicted class", CLASS_NAMES[best], f"{float(scores[best])*100:.1f}%")
                for i in top:
                    i = int(i)
                    st.write(f"**{CLASS_NAMES[i]}** — {float(scores[i])*100:.2f}%")
                    st.progress(float(np.clip(scores[i], 0.0, 1.0)))
        except Exception as exc:
            st.exception(exc)

st.divider()
st.caption(
    "CIFAR-10 contains only 10 categories. This model always chooses among those "
    "categories; it is not an open-set detector. Scores are model outputs, not "
    "calibrated probabilities."
)
