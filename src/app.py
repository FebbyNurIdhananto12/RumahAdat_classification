import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shap  # For SHAP interpretation
from pathlib import Path

# Title of the Streamlit application
st.title("Mengidentifikasi Jenis Tanaman Bunga Hias")

# Model selection
model_choice = st.selectbox("Pilih Model:", ["model_vgg.h5", "mobilenetv3_rumah_adat"])

# Function to load and preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert("RGB")
    img = img.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to perform prediction with the selected model
def predict_image(img, model_path):
    class_names = ['gadang', 'honai', 'joglo', 'panjang', 'tongkonan']
    img_array = preprocess_image(img)
    
    # Load the selected model
    model = tf.keras.models.load_model(model_path)
    
    # Make predictions
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    probability = np.max(tf.nn.softmax(prediction[0]))
    
    return predicted_class, probability

# Function to visualize the prediction results
def visualize_prediction(img, predicted_class, probability):
    st.image(img, caption="Gambar yang diunggah", use_container_width=True)
    st.subheader("Hasil prediksi:")
    st.write(f"Prediksi: {predicted_class}")
    st.write(f"Probabilitas: {probability * 100:.2f}%")

# Upload image section
upload = st.file_uploader("Unggah gambar (jpg, png, jpeg)", type=['jpg', 'png', 'jpeg'])

if st.button("Prediksi", type="primary"):
    if upload is not None:
        with st.spinner('Memproses gambar untuk prediksi...'):
            model_path = f"C:/Users/Febby/Documents/OneDrive/Documents/Kuliah/Semester 7/Machine Learning/BISSMILLAHHH/src/{model_choice}"
            predicted_class, probability = predict_image(upload, model_path)
            visualize_prediction(upload, predicted_class, probability)
            
            # Optional: SHAP interpretation
            if st.checkbox("Tampilkan Interpretasi Model dengan SHAP"):
                explainer = shap.KernelExplainer(model.predict, preprocess_image(upload))
                shap_values = explainer.shap_values(preprocess_image(upload))
                shap.image_plot(shap_values, preprocess_image(upload))
    else:
        st.write("Unggah gambar terlebih dahulu!")