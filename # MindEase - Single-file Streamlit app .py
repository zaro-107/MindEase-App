# MindEase - Single-file Streamlit app (starter)
# Features:
# - Face emotion detection (using uploaded photo or webcam via st.camera_input)
# - Speech emotion detection (upload .wav) using librosa features
# - Simple recommendation system based on predicted emotion
# - Emotion trend tracking saved locally in mood_log.json
# Notes:
# - This is a starter app. For production, replace placeholder models with real trained models.
# - Put pretrained models (Keras .h5) in the same folder named:
#     face_emotion_model.h5
#     audio_emotion_model.h5
# - Haar cascade for face detection is optional; the code ships a fallback.

import streamlit as st
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
import tempfile

# Image and audio processing
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import librosa

# Visualization
import matplotlib.pyplot as plt

# ------------------------- Configuration -------------------------
LOG_FILE = "mood_log.json"
HAAR_PATH = "haarcascade_frontalface_default.xml"  # optional: drop this file here
FACE_MODEL_PATH = "face_emotion_model.h5"
AUDIO_MODEL_PATH = "audio_emotion_model.h5"

# Emotions used across models (ensure consistency)
EMOTIONS = ["neutral", "happy", "sad", "angry", "surprise", "fear", "disgust"]

# Recommendation library (simple mapping)
RECOMMENDATIONS = {
    "happy": ["Keep it up! Play a celebratory playlist", "Share your joy with a friend", "Short gratitude journaling - 3 things"],
    "sad": ["Try 5 minutes of deep breathing (4-4-8)", "Listen to a calming playlist", "Try a gentle walk outside"],
    "angry": ["Do 3 grounding breaths; 10 pushups or physical release", "Try progressive muscle relaxation", "Step away for 5 minutes and decompress"],
    "surprise": ["Take a moment to process the surprise", "Journal what surprised you and why"],
    "fear": ["Try box breathing (4-4-4-4)", "Play calming music", "Use grounding technique: name 5 things you see"],
    "disgust": ["Take a few slow breaths and reframe the situation", "Distract with a pleasant sensory activity"],
    "neutral": ["Explore a micro-meditation", "Listen to focus music", "Try a short mindful stretch"]
}

# ---------------------- Utilities ----------------------

def ensure_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            json.dump([], f)


def append_log(entry: dict):
    ensure_log()
    with open(LOG_FILE, "r+") as f:
        data = json.load(f)
        data.append(entry)
        f.seek(0)
        json.dump(data, f, indent=2)


def load_log():
    ensure_log()
    with open(LOG_FILE, "r") as f:
        return json.load(f)


# ---------------------- Face processing ----------------------

def load_face_model(path=FACE_MODEL_PATH):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.warning(f"Could not load face model from {path}: {e}. Using fallback heuristic.")
        return None


def detect_face_and_predict(image_bytes, model=None):
    # image_bytes: PIL file-like or bytes from st.camera_input
    # Convert to OpenCV image
    import numpy as np
    from PIL import Image
    if isinstance(image_bytes, bytes):
        img = Image.open(tempfile.SpooledTemporaryFile().write(image_bytes))
    else:
        img = Image.open(image_bytes)
    img = img.convert('RGB')
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Try to load Haar cascade
    face_cascade = None
    if os.path.exists(HAAR_PATH):
        face_cascade = cv2.CascadeClassifier(HAAR_PATH)

    faces = []
    if face_cascade is not None:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # fallback: use center crop
        h, w = gray.shape
        cx, cy = w // 2, h // 2
        size = int(min(w, h) * 0.6)
        x1 = max(0, cx - size // 2)
        y1 = max(0, cy - size // 2)
        face_img = open_cv_image[y1:y1+size, x1:x1+size]
    else:
        # pick largest face
        faces = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)
        (x, y, w_, h_) = faces[0]
        face_img = open_cv_image[y:y+h_, x:x+w_]

    # Preprocess for model
    face_resized = cv2.resize(face_img, (48, 48))  # common size
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    face_gray = face_gray.astype('float') / 255.0
    face_gray = img_to_array(face_gray)
    face_gray = np.expand_dims(face_gray, axis=0)

    if model is not None:
        preds = model.predict(face_gray)
        idx = int(np.argmax(preds))
        emotion = EMOTIONS[idx] if idx < len(EMOTIONS) else "neutral"
        confidence = float(np.max(preds))
    else:
        # simple heuristic: mean pixel brightness -> happy/neutral/sad
        mean_val = face_gray.mean()
        if mean_val > 0.6:
            emotion = "happy"
        elif mean_val < 0.35:
            emotion = "sad"
        else:
            emotion = "neutral"
        confidence = 0.6

    return emotion, confidence, face_img


# ---------------------- Audio processing ----------------------

def load_audio_model(path=AUDIO_MODEL_PATH):
    try:
        model = load_model(path)
        return model
    except Exception as e:
        st.warning(f"Could not load audio model from {path}: {e}. Using fallback heuristic.")
        return None


def extract_audio_features(file_path, sr=22050, n_mfcc=40):
    y, sr = librosa.load(file_path, sr=sr, mono=True, duration=5.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled


def predict_audio_emotion(file_path, model=None):
    feats = extract_audio_features(file_path)
    feats = np.expand_dims(feats, axis=0)  # shape (1, n_mfcc)
    if model is not None:
        preds = model.predict(feats)
        idx = int(np.argmax(preds))
        emotion = EMOTIONS[idx] if idx < len(EMOTIONS) else "neutral"
        confidence = float(np.max(preds))
    else:
        # fallback: energy-based heuristic
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=5.0)
        energy = np.sum(y ** 2) / len(y)
        if energy > 0.01:
            emotion = "angry"
        else:
            emotion = "sad"
        confidence = 0.6
    return emotion, confidence


# ---------------------- Recommendation ----------------------

def recommend_for_emotion(emotion):
    return RECOMMENDATIONS.get(emotion, RECOMMENDATIONS["neutral"])[:3]


# ---------------------- Streamlit UI ----------------------

st.set_page_config(page_title="MindEase — Emotion-Aware Wellness", layout="centered")
st.title("MindEase — Emotion-Aware Mental Wellness App (Starter)")
st.markdown("Your AI companion that listens, understands, and heals — powered by emotion recognition.")

# Load models (best-effort)
with st.spinner("Loading models (if available)..."):
    face_model = load_face_model()
    audio_model = load_audio_model()

col1, col2 = st.columns(2)

with col1:
    st.header("Face Emotion")
    st.write("Use your webcam (camera) or upload a photo.")
    img_file = st.camera_input("Take a selfie")
    uploaded_img = st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])

    chosen_image = None
    if img_file is not None:
        chosen_image = img_file
    elif uploaded_img is not None:
        chosen_image = uploaded_img

    if chosen_image is not None:
        try:
            emotion, conf, face_img = detect_face_and_predict(chosen_image, face_model)
            st.subheader(f"Detected emotion: {emotion}  (confidence: {conf:.2f})")
            st.image(face_img, caption="Detected face (cropped)")
            recs = recommend_for_emotion(emotion)
            st.markdown("**Suggestions:**")
            for r in recs:
                st.write(f"- {r}")

            # Log
            entry = {
                "source": "face",
                "emotion": emotion,
                "confidence": conf,
                "timestamp": datetime.utcnow().isoformat()
            }
            append_log(entry)
        except Exception as e:
            st.error(f"Error processing image: {e}")

with col2:
    st.header("Speech Emotion")
    st.write("Upload a short audio clip (.wav recommended)")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg"])

    if audio_file is not None:
        try:
            # save temporarily
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(audio_file.read())
            tfile.flush()
            audio_path = tfile.name

            emotion, conf = predict_audio_emotion(audio_path, audio_model)
            st.subheader(f"Detected emotion: {emotion}  (confidence: {conf:.2f})")
            recs = recommend_for_emotion(emotion)
            st.markdown("**Suggestions:**")
            for r in recs:
                st.write(f"- {r}")

            entry = {
                "source": "audio",
                "emotion": emotion,
                "confidence": conf,
                "timestamp": datetime.utcnow().isoformat()
            }
            append_log(entry)

            # remove temp file
            try:
                os.unlink(audio_path)
            except:
                pass
        except Exception as e:
            st.error(f"Error processing audio: {e}")

st.markdown("---")

# Emotion trends
st.header("Emotion Trends & Insights")
if st.button("Load trend data"):
    data = load_log()
    if len(data) == 0:
        st.info("No mood data logged yet. Use Face or Speech detection to add entries.")
    else:
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        st.write(df.tail(20))

        # simple counts over time
        counts = df.groupby([pd.Grouper(key='timestamp', freq='D'), 'emotion']).size().unstack(fill_value=0)
        st.line_chart(counts)

        # show distribution
        dist = df['emotion'].value_counts()
        fig, ax = plt.subplots()
        dist.plot(kind='bar', ax=ax)
        ax.set_ylabel('Count')
        ax.set_title('Emotion distribution')
        st.pyplot(fig)

st.markdown("---")

st.header("Developer Notes & Next Steps")
st.markdown(
"""
- This is a starter single-file app. Replace the heuristic fallbacks with properly trained models:
  - Face: train a CNN on FER2013 or similar and save as `face_emotion_model.h5` with the same EMOTIONS order.
  - Audio: train an audio classifier on RAVDESS / CREMA-D / SAVEE etc., save as `audio_emotion_model.h5`.
- Add authentication if storing sensitive mood data.
- Improve face detection with modern detectors (MTCNN / dlib / mediapipe).
- For real-time face webcam stream, consider using `streamlit-webrtc`.

Run the app:
    pip install -r requirements.txt
    streamlit run mindease_app.py

Requirements example (put in requirements.txt):
streamlit
tensorflow
opencv-python
librosa
matplotlib
pandas

"""
)

st.write("App created by MindEase starter template — customize models and UX as needed.")
