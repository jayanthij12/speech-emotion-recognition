import numpy as np
import librosa
import joblib

from tensorflow.keras.models import load_model


# ----------------------------
# Load CNN Model
# ----------------------------

cnn_model = load_model("models/cnn_model.keras")

label_encoder = joblib.load(
    "models/cnn_label_encoder.pkl"
)


# ----------------------------
# MFCC Extraction
# ----------------------------

def extract_mfcc(file_path):

    signal, sample_rate = librosa.load(
        file_path,
        sr=None
    )

    signal = librosa.util.fix_length(
        signal,
        size=22050 * 3
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    return mfcc


# ----------------------------
# Prediction
# ----------------------------

def predict_cnn(file_path):

    mfcc = extract_mfcc(file_path)

    mfcc = np.expand_dims(
        mfcc,
        axis=-1
    )

    mfcc = np.expand_dims(
        mfcc,
        axis=0
    )

    mfcc = mfcc / np.max(np.abs(mfcc))

    probabilities = cnn_model.predict(
        mfcc,
        verbose=0
    )[0]

    emotion = label_encoder.inverse_transform(
        [np.argmax(probabilities)]
    )[0]

    return emotion, probabilities