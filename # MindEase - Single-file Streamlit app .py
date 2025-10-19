# MindEase - Advanced Streamlit App

import streamlit as st
import cv2
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tempfile
import os

st.set_page_config(page_title="MindEase AI", page_icon="🧠", layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("MindEase Controls")
st.sidebar.write("Upload your files and get predictions.")

# Tabs for UI
tab1, tab2, tab3 = st.tabs(["Image Analysis", "Audio Analysis", "About"])

# ---------------- Image Analysis ----------------
with tab1:
    st.header("🖼️ Image Emotion Detection")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"], key="image")

    if uploaded_image:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Preprocess and predict
        model_image = load_model("models/image_model.h5")  # replace with your model
        img_resized = cv2.resize(image, (224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model_image.predict(img_array)[0]
        classes = ["Happy", "Sad", "Neutral", "Angry", "Surprised"]  # example categories

        max_idx = np.argmax(prediction)
        st.success(f"Prediction: **{classes[max_idx]}**")
        st.info(f"Confidence: {prediction[max_idx]*100:.2f}%")

# ---------------- Audio Analysis ----------------
with tab2:
    st.header("🎵 Audio Emotion Detection")
    uploaded_audio = st.file_uploader("Upload audio", type=["wav", "mp3"], key="audio")

    if uploaded_audio:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_audio.read())
            audio_path = tmp_file.name

        st.audio(audio_path, format='audio/wav')

        # Load audio and extract features
        y, sr = librosa.load(audio_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0).reshape(1, -1)

        model_audio = load_model("models/audio_model.h5")  # replace with your model
        prediction_audio = model_audio.predict(mfccs_processed)[0]
        classes_audio = ["Happy", "Sad", "Neutral", "Angry", "Fear"]  # example categories

        max_idx_audio = np.argmax(prediction_audio)
        st.success(f"Prediction: **{classes_audio[max_idx_audio]}**")
        st.info(f"Confidence: {prediction_audio[max_idx_audio]*100:.2f}%")

        # Plot waveform
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Audio Waveform")
        st.pyplot(fig)

        # Plot spectrogram
        fig2, ax2 = plt.subplots()
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax2)
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")
        ax2.set_title("Spectrogram")
        st.pyplot(fig2)

# ---------------- About ----------------
with tab3:
    st.header("ℹ️ About MindEase AI")
    st.write("""
        MindEase AI is an advanced mental health assistant using computer vision and audio analysis.
        - Detects emotions from images and audio.
        - Provides confidence scores.
        - Helps users track mental wellness.
        """)
