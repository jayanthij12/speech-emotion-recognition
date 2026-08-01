import joblib

from feature_extractor import extract_features


model = joblib.load("models/best_emotion_model.pkl")
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")


def predict_emotion(audio_path):

    features = extract_features(audio_path)

    features = scaler.transform(features)

    prediction = model.predict(features)

    emotion = label_encoder.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(features)[0]

    return emotion, probabilities