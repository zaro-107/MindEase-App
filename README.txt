# MindEase — Emotion-Aware Mental Wellness App

## Overview

MindEase is a Streamlit-based AI application that detects your emotions through **face recognition** or **speech analysis**, and provides **personalized stress-relief suggestions** such as calming music, breathing exercises, or mini-therapy prompts. It also tracks emotional trends over time and generates mental health insights.

**Tagline:** *“Your AI companion that listens, understands, and heals — powered by emotion recognition.”*

---

## Features

* **Face Emotion Detection**: Upload a selfie or use your webcam to detect your current emotion.
* **Speech Emotion Detection**: Upload a short audio clip to analyze your voice for emotions.
* **Personalized Recommendations**: Get mood-based activity suggestions like breathing exercises, walks, or calming playlists.
* **Emotion Trend Tracking**: Logs your emotions over time and displays visual trends and distribution.
* **Extensible**: Replace heuristic models with pretrained CNN/RNN models for better accuracy.

---

## Tech Stack

* **Frontend & Deployment**: Streamlit
* **Deep Learning Models**: TensorFlow / Keras
* **Image Processing**: OpenCV
* **Audio Processing**: Librosa
* **Data Handling & Visualization**: Pandas, Numpy, Matplotlib, Seaborn
* **Model Evaluation**: Scikit-learn
* **Storage**: Local JSON file (`mood_log.json`) for logging

Optional additions for production:

* GUI enhancements: `streamlit-webrtc` for real-time webcam feed
* Cloud database storage: Firebase / MongoDB
* Voice assistant: `pyttsx3` / `speechrecognition`

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/MindEase.git
cd MindEase
```

2. Install the required Python packages:

```bash
pip install -r requirements.txt
```

3. Add pretrained models (optional for better accuracy):

* `face_emotion_model.h5`
* `audio_emotion_model.h5`

4. Add Haar cascade file for face detection (optional):

* `haarcascade_frontalface_default.xml`

---

## Running the App Locally

```bash
streamlit run mindease_app.py
```

* Open the browser link shown in the terminal (usually `http://localhost:8501`).
* Grant camera/microphone access if using real-time face/audio detection.

---

## Deployment (Streamlit Cloud)

1. Push the repository to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io) and log in.
3. Click **New App**, select your repo and branch, set `mindease_app.py` as the main file.
4. Click **Deploy**.
5. Access your app via the public URL provided.

---

## Folder Structure

```
MindEase/
│
├── mindease_app.py         # Main Streamlit app
├── requirements.txt        # Dependencies
├── face_emotion_model.h5   # Optional pretrained model
├── audio_emotion_model.h5  # Optional pretrained model
├── haarcascade_frontalface_default.xml  # Optional face detector
└── mood_log.json           # Emotion tracking log (auto-created)
```

---

## Next Steps / Improvements

* Train accurate CNN for face emotion detection on FER2013 or similar dataset.
* Train RNN/LSTM for speech emotion recognition using RAVDESS, CREMA-D, or SAVEE.
* Implement real-time emotion detection with `streamlit-webrtc`.
* Add cloud storage for logs to access from multiple devices.
* Enhance UI/UX with richer recommendations and visualization.

---

## License

MIT License © 2025 Pradhuman Singh Shekhawat
