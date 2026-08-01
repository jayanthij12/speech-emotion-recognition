import numpy as np
import librosa


def extract_features(file_path):

    signal, sample_rate = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=13
    )
    mfcc = np.mean(mfcc.T, axis=0)

    chroma = librosa.feature.chroma_stft(
        y=signal,
        sr=sample_rate
    )
    chroma = np.mean(chroma.T, axis=0)

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate
    )
    mel = np.mean(mel.T, axis=0)

    zcr = librosa.feature.zero_crossing_rate(signal)
    zcr = np.mean(zcr.T, axis=0)

    rms = librosa.feature.rms(y=signal)
    rms = np.mean(rms.T, axis=0)

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sample_rate
    )
    centroid = np.mean(centroid.T, axis=0)

    features = np.hstack([
        mfcc,
        chroma,
        mel,
        zcr,
        rms,
        centroid
    ])

    return features.reshape(1, -1)