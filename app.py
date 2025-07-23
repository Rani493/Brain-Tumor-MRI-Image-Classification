import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# --- THIS BLOCK MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="Brain MRI Tumor Classifier",
    page_icon="🧠",
    layout="centered"
)
# --- END OF set_page_config BLOCK ---

# --- Configuration (can be here) ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
MODEL_PATH = './saved_model/my_brain_mri_model.keras'
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# --- Load the Model (cached function, also fine here) ---
@st.cache_resource
def load_trained_model():
    """Loads the pre-trained Keras model."""
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model. Please check the model path and file: '{MODEL_PATH}'. Error details: {e}")
        st.stop()

model = load_trained_model()

# --- Preprocessing Function (function definition, fine here) ---
def preprocess_image(image):
    """
    Preprocesses the uploaded image to match the model's input requirements.
    Resizes, ensures RGB format, converts to numpy array, and normalizes pixel values.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((IMG_HEIGHT, IMG_WIDTH))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- Streamlit App Content (starts AFTER set_page_config) ---
st.title("🧠 Brain MRI Tumor Classifier")
st.markdown("""
Upload a brain MRI scan image (JPG, JPEG, or PNG) and the model will predict the type of tumor.
""")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)
    st.write("")

    with st.spinner('Classifying...'):
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

    st.success(f"Prediction: **{predicted_class_name}**")
    st.info(f"Confidence: **{confidence:.2f}%**")

    st.markdown("---")
    st.subheader("All Class Probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"- {class_name}: {predictions[0][i]*100:.2f}%")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and TensorFlow")