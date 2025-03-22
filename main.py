import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the trained model
model = tf.keras.models.load_model('Model.h5')

# Define cancer type labels
labels = ['Colon Adenocarcinoma', 'Colon Benign Tissue', 'Lung Adenocarcinoma', 'Lung Benign Tissue', 'Lung Squamous Cell Carcinoma']

def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Adjust size as per model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

def predict_cancer(img_array):
    predictions = model.predict(img_array)[0]  # Get first row if batch size is 1
    return {labels[i]: float(predictions[i]) for i in range(len(labels))}

def plot_pie_chart(predictions):
    fig, ax = plt.subplots()
    ax.pie(predictions.values(), labels=predictions.keys(), autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    st.pyplot(fig)

# Streamlit UI
st.title("Cancer Type Prediction")
uploaded_file = st.file_uploader("Upload an image for prediction", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    img_array = preprocess_image(uploaded_file)
    predictions = predict_cancer(img_array)
    
    st.subheader("Prediction Probabilities:")
    for label, probability in predictions.items():
        st.write(f"{label}: {probability*100:.2f}%")
    
    plot_pie_chart(predictions)
