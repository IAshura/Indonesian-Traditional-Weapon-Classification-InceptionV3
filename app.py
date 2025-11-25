# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import os
import matplotlib.pyplot as plt

# =========================================
# Streamlit Config
# =========================================
st.set_page_config(
    page_title="Indonesian Traditional Weapon Classifier",
    page_icon="⚔️",
    layout="centered"
)

# =========================================
# Constants / Labels
# =========================================
# class names for prediction
CLASS_NAMES = [
    "Busur", "Celurit", "Golok", "Kerambit",
    "Keris", "Kudi", "Plinteng", "Tombak", "Wedhung"
]

# short labels for bar charts (option B requested)
SHORT_LABELS = [
    "R-Adam", "R-Adamax", "R-RMS", "R-SGD",
    "RA-Adam", "RA-Adamax", "RA-RMS", "RA-SGD"
]

# accuracy data as provided
VALIDATION_ACCS = [0.90, 0.86, 0.91, 0.39, 0.96, 0.89, 0.96, 0.73]
TEST_ACCS       = [0.92, 0.82, 0.92, 0.42, 0.96, 0.91, 0.96, 0.69]

# sample image path (from your session files)
# developer note: using a local path from the session history so you can preview without uploading
SAMPLE_IMAGE_PATH = "/mnt/data/a0104c75-8027-436c-95f5-ee2e895de07c.png"

# =========================================
# Utility functions
# =========================================
@st.cache_resource
def load_model(path):
    """Try to load a keras model. Return model or Exception object."""
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        return e

def preprocess_image(image, target_size=(299, 299)):
    """PIL image -> preprocessed numpy array (InceptionV3 style)."""
    img = image.convert("RGB").resize(target_size)
    arr = tf.keras.preprocessing.image.img_to_array(img)
    arr = tf.keras.applications.inception_v3.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def build_model_path(augment_folder, optimizer):
    """
    Build path like:
    results/Model/{Dataset_R or Dataset_R_A}/{Optimizer}/Model_InceptionV3/model.h5
    """
    base = "results\Model"
    return os.path.join(base, augment_folder, optimizer, "Model_InceptionV3", "model.h5")

# =========================================
# Sidebar: Model selection (augment + optimizer)
# =========================================
st.sidebar.header("Model Settings")

# Augmentation option
augment_option = st.sidebar.selectbox(
    "Augmentation",
    options=["No augmentation", "With augmentation"],
    format_func=lambda x: x
)
# Map to folder names used in your filesystem
if augment_option == "With augmentation":
    augment_folder = "Dataset_R_A"
else:
    augment_folder = "Dataset_R"

# Optimizer option
optimizer_option = st.sidebar.selectbox(
    "Optimizer",
    options=["Adam", "Adamax", "RMSprop", "SGD"]
)

# Construct model path automatically
auto_model_path = build_model_path(augment_folder, optimizer_option)

# Allow user to override if needed
st.sidebar.markdown("**Auto model path**:")
st.sidebar.code(auto_model_path)
model_path = st.sidebar.text_input("Or enter model path (.h5) manually", value=auto_model_path)

st.sidebar.write("---")
st.sidebar.caption("If model load fails, check the path above or use a different combination.")

# Button to (re)load model
if st.sidebar.button("Load Model"):
    # Clear cache of load_model before reloading on demand (optional)
    try:
        load_model.__wrapped__(model_path)  # call underlying function once to refresh cache if needed
    except Exception:
        pass

# =========================================
# Load model (show spinner and error handling)
# =========================================
with st.spinner("Loading model..."):
    model_loaded = load_model(model_path)

if isinstance(model_loaded, Exception):
    st.sidebar.error("Failed to load model. Check path or model file.")
    st.error(f"Error loading model:\n{model_loaded}")
    # Do not stop the app entirely — allow user to still view charts or try sample image
else:
    st.sidebar.success("Model loaded successfully ✔️")

# =========================================
# Page title / instructions
# =========================================
st.title("⚔️ Indonesian Traditional Weapon Classification")
st.markdown(
    "Upload an image of a traditional Indonesian weapon or use the sample image. "
    "Choose augmentation and optimizer in the sidebar to load a specific trained model."
)
st.write("---")

# =========================================
# Combined Double Bar Chart (Validation + Test)
# =========================================
st.header("Model Performance Overview")

fig, ax = plt.subplots(figsize=(10, 4))

x = np.arange(len(SHORT_LABELS))
width = 0.35  # gap antar batang

# dua bar tanpa warna spesifik (Streamlit rules)
ax.bar(x - width/2, VALIDATION_ACCS, width, label='Validation')
ax.bar(x + width/2, TEST_ACCS, width, label='Test')

ax.set_xticks(x)
ax.set_xticklabels(SHORT_LABELS, rotation=45, ha="right")
ax.set_ylim(0, 1.0)
ax.set_ylabel("Accuracy")
ax.set_title("Validation vs Test Accuracy")
ax.legend()

fig.tight_layout()
st.pyplot(fig)


st.write("---")

# =========================================
# Image upload / sample + prediction area
# Layout: left image, right prediction
# =========================================
st.subheader("🔎 Try the model")

# file uploader and sample button
uploaded = st.file_uploader("📤 Upload Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

image = None
if uploaded:
    try:
        image = Image.open(io.BytesIO(uploaded.read()))
    except UnidentifiedImageError:
        st.error("Invalid image file.")
        st.stop()
else:
    st.info("Upload an image to start the classification.")

# If image provided, show layout and attempt prediction if model loaded
if image is not None:
    img_col, pred_col = st.columns([1, 1])

    # show image (left)
    with img_col:
        st.image(image, caption="Uploaded Image", width=300)
        st.caption(f"Image size: {image.size[0]}×{image.size[1]} px")

    # prediction (right)
    with pred_col:
        st.subheader("Prediction:")
        if isinstance(model_loaded, Exception):
            st.warning("Model not loaded — cannot run prediction. Fix model path in sidebar.")
        else:
            try:
                x = preprocess_image(image)
                preds = model_loaded.predict(x)
                probs = preds[0]
                top_idx = int(np.argmax(probs))
                top_class = CLASS_NAMES[top_idx]
                top_conf = float(probs[top_idx]) * 100
            except Exception as e:
                st.error(f"Prediction failed:\n{e}")
                probs = None
                top_class, top_conf = None, None

            if probs is not None:
                # display main result left-aligned
                st.markdown(
                    f"<h3 style='text-align:left;'>This is: <b>{top_class.upper()}</b> ({top_conf:.0f}%)</h3>",
                    unsafe_allow_html=True
                )
                # progress bar
                st.progress(float(top_conf) / 100)

                # top 3
                st.markdown("#### Top 3 Predictions :")
                top3 = probs.argsort()[-3:][::-1]
                for i in top3:
                    pct = probs[i] * 100
                    st.write(f"- **{CLASS_NAMES[i]}**: {pct:.2f}%")

                # expandable full probabilities with nicer display (progress bars)
                with st.expander("📊 Show All Class Probabilities"):
                    # show each as text + small horizontal bar
                    for i, p in enumerate(probs):
                        pct = p * 100
                        st.write(f"{i+1}. **{CLASS_NAMES[i]}** — {pct:.2f}%")
                        st.progress(float(p))


# =========================================
# Footer / tips
# =========================================
st.write("---")
st.caption("Tip: choose Augmentation and Optimizer on the left to load a different trained model. "
           "Model path auto-built but you can override it manually if your structure differs.")
