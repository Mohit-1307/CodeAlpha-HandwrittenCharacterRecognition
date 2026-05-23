import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered"
)

# =========================
# Load Trained Model
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# =========================
# Title
# =========================

st.title("✍️ Handwritten Character Recognition")

st.write(
    "Upload a handwritten digit image "
    "and the CNN model will predict it."
)

# =========================
# Upload Image
# =========================

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

# =========================
# Prediction Pipeline
# =========================

if uploaded_file is not None:

    try:

        # Load image
        image = Image.open(uploaded_file).convert("L")

        # Convert to numpy array
        img = np.array(image)

        # Resize
        img = cv2.resize(img, (28, 28))

        # Invert colors
        img = 255 - img

        # Normalize
        img = img.astype("float32") / 255.0

        # Save processed image for display
        processed_img = img.copy()

        # Reshape for CNN
        img = img.reshape(1, 28, 28, 1)

        # Prediction
        prediction = model.predict(img)

        predicted_class = np.argmax(prediction)

        confidence = np.max(prediction)

        # =========================
        # Display Images
        # =========================

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:
            st.image(
                processed_img,
                caption="Processed Image",
                use_container_width=True
            )

        # =========================
        # Results
        # =========================

        st.success(
            f"Predicted Digit: {predicted_class}"
        )

        st.info(
            f"Confidence: {confidence:.4f}"
        )

    except Exception as e:

        st.error(f"Error processing image: {e}")