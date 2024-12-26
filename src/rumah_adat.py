import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Set the categories for prediction
CATEGORIES = ['gadang', 'honai', 'joglo', 'panjang', 'tongkonan']

# Load models
@st.cache_resource
def load_vgg_model():
    # Gantilah 'path_to_vgg_model.h5' dengan path yang benar
    return load_model("C:/Users/Febby/Documents/OneDrive/Documents/Kuliah/Semester 7/Machine Learning/BISSMILLAHHH/src/model_vgg.h5")  # Misalnya ada di folder 'models'

@st.cache_resource
def load_mobilenet_model():
    # Gantilah 'path_to_mobilenet_model.h5' dengan path yang benar
    return load_model("C:/Users/Febby/Documents/OneDrive/Documents/Kuliah/Semester 7/Machine Learning/BISSMILLAHHH/src/mobilenetv3_rumah_adat.h5")  # Misalnya ada di folder 'models'

# Prediction function
def predict_image(model, image):
    # Mengonversi gambar ke RGB jika gambar memiliki 4 saluran warna (RGBA)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Mengubah ukuran gambar menjadi 150x150 piksel (sesuai dengan input model)
    image = image.resize((150, 150))
    
    # Menormalisasi gambar dan mengubah dimensinya
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    
    # Melakukan prediksi
    predictions = model.predict(img_array)[0]
    return predictions

# Streamlit app
st.set_page_config(page_title="Rumah Adat Classifier", layout="wide")

st.title("Rumah Adat Classifier")
st.write("Aplikasi ini memungkinkan Anda memprediksi jenis rumah adat dari gambar menggunakan model VGG atau MobileNet.")

# Sidebar for model selection
st.sidebar.title("Pengaturan Model")
model_choice = st.sidebar.selectbox("Pilih Model", ["VGG", "MobileNet"])
model = load_vgg_model() if model_choice == "VGG" else load_mobilenet_model()

# File uploader for image input
uploaded_file = st.file_uploader("Unggah gambar rumah adat", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    
    # Predict button
    if st.button("Prediksi"):
        with st.spinner("Sedang memproses..."):
            predictions = predict_image(model, image)
            max_index = np.argmax(predictions)
            predicted_label = CATEGORIES[max_index]
            confidence = predictions[max_index] * 100

            if predicted_label in CATEGORIES:
                st.success(f"Prediksi: **{predicted_label.capitalize()}** dengan kepercayaan {confidence:.2f}%")
            else:
                st.warning("Model hanya dapat memprediksi rumah adat dari kategori berikut:")
                st.write(", ".join(CATEGORIES))
