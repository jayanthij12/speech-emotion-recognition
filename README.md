# 🎤 EmotionSense AI Pro

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CNN-FF6F00?logo=tensorflow)
![License](https://img.shields.io/badge/License-Educational-green)

</p>

An end-to-end **Speech Emotion Recognition (SER)** system that predicts human emotions from speech audio using both **Machine Learning (Random Forest)** and **Deep Learning (CNN)**. The application provides an interactive dashboard with rich visualizations, confidence analysis, prediction history, downloadable reports, and AI-generated emotion interpretation.

---

# 📌 Project Overview

EmotionSense AI Pro analyzes a user's speech recording and predicts the underlying emotional state using acoustic signal processing and machine learning techniques.

The application extracts audio features, performs emotion classification using either a **Random Forest** model or a **Convolutional Neural Network (CNN)**, and presents the results through a professional Streamlit dashboard.

---

# ✨ Features

### 🎤 Audio Processing

- Upload WAV audio files
- Built-in audio player
- Automatic audio preprocessing
- Audio statistics extraction

### 🤖 Emotion Recognition

- Random Forest Classifier
- CNN Deep Learning Model
- 8 Emotion Classification
- Confidence Score
- Top-3 Predictions
- Probability Distribution

### 📊 Interactive Dashboard

- Prediction Result Card
- Model Confidence Panel
- AI Emotion Interpretation
- Model Information
- Audio Information
- Prediction History
- Download Prediction Report

### 📈 Audio Visualizations

- Waveform
- Spectrogram
- MFCC Heatmap
- Emotion Probability Chart
- Probability Table

---

# 🧠 Supported Emotions

| Emotion | Emoji |
|---------|-------|
| Angry | 😡 |
| Calm | 😌 |
| Disgust | 🤢 |
| Fearful | 😨 |
| Happy | 😊 |
| Neutral | 😐 |
| Sad | 😢 |
| Surprised | 😲 |

---

# 🧠 Machine Learning Pipeline

```text
Speech Audio (.wav)
        │
        ▼
Audio Preprocessing
        │
        ▼
Feature Extraction
(MFCC + Chroma + Mel + ZCR + RMS + Spectral Centroid)
        │
        ▼
Feature Scaling
        │
        ▼
Emotion Classification
(Random Forest / CNN)
        │
        ▼
Probability Estimation
        │
        ▼
Interactive Dashboard
```

---

# 🛠️ Technologies Used

## Programming

- Python

## Machine Learning

- Scikit-Learn
- TensorFlow / Keras

## Audio Processing

- Librosa
- NumPy
- SciPy

## Data Processing

- Pandas

## Visualization

- Plotly
- Matplotlib

## Web Application

- Streamlit

---

# 📂 Dataset

**Dataset Used**

**RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)**

Dataset Characteristics

- 8 Emotion Classes
- High-quality Speech Recordings
- Professional Actors
- Balanced Dataset
- WAV Audio Format

---

# 🤖 Models Used

## Machine Learning

### Random Forest Classifier

- Feature-based classification
- Fast inference
- Probability prediction
- StandardScaler preprocessing

### Feature Vector (156 Features)

- 13 MFCC
- 12 Chroma
- 128 Mel Spectrogram
- Zero Crossing Rate
- RMS Energy
- Spectral Centroid

---

## Deep Learning

### Convolutional Neural Network (CNN)

Input Shape

```
40 × 130 MFCC
```

Architecture

- Convolution Layers
- Max Pooling
- Dropout
- Dense Layers
- Softmax Output

---

# Home
![Home](assets/application_screenshots/home_page.jpg)


# 📁 Project Structure

```text
EmotionSense_AI/
│
├── app/
│   ├── streamlit_app.py
│   ├── predictor.py
│   ├── cnn_predictor.py
│   └── feature_extractor.py
│
├── assets/
│   ├── application_screenshots/
│   └── notebook_screenshots/
│
├── datasets/
├── models/
├── notebooks/
│
├── outputs/
├── reports/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/yourusername/EmotionSense_AI.git
```

Navigate to the project

```bash
cd EmotionSense_AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/streamlit_app.py
```

---

# 📋 How to Use

1. Launch the Streamlit application.
2. Choose a prediction model:
   - Random Forest
   - CNN (Deep Learning)
3. Upload a WAV audio file.
4. Click **Predict Emotion**.
5. View:
   - Predicted Emotion
   - Confidence Score
   - AI Interpretation
   - Audio Statistics
   - Waveform
   - Spectrogram
   - MFCC Heatmap
   - Top-3 Predictions
   - Probability Chart
   - Prediction History
6. Download the prediction report.

---

# 📊 Outputs

The application generates:

- Emotion Prediction
- Confidence Analysis
- Model Information
- Audio Information
- AI Interpretation
- Audio Waveform
- Spectrogram
- MFCC Heatmap
- Top-3 Predictions
- Probability Chart
- Probability Table
- Prediction History
- Downloadable Report

---

# 🔮 Future Improvements

- Real-time microphone emotion recognition
- Transformer-based speech models
- Multi-language emotion recognition
- Speech-to-text integration
- Model explainability (SHAP/LIME)
- Cloud deployment
- REST API
- User authentication
- Emotion analytics dashboard

---

# 👩‍💻 Developer

**Jakkula Jayanthi**

Machine Learning Intern

CodeAlpha Internship Project

---

# 📜 License

This project is developed for educational, research, and internship purposes.

The RAVDESS dataset is used solely for academic and non-commercial use in accordance with its licensing terms.

---

⭐ If you found this project useful, consider giving it a star on GitHub!
