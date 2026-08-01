import streamlit as st
import tempfile
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime

from predictor import predict_emotion
from cnn_predictor import predict_cnn

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="EmotionSense AI Pro",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

with st.sidebar:

    st.title("🎤 EmotionSense AI Pro")

    st.caption(
        "Speech Emotion Recognition using Machine Learning and Deep Learning"
    )

    st.success("✅ Model Loaded Successfully")

    st.markdown("---")

    st.subheader("🤖 Prediction Model")

    selected_model = st.radio(
        "Choose Model",
        [
            "Random Forest",
            "CNN (Deep Learning)"
        ]
    )

    st.success(f"Selected Model: {selected_model}")

    st.markdown("---")

    st.subheader("Supported Emotions")

    st.markdown("""
        😊 Happy

        😐 Neutral

        😌 Calm

        😢 Sad

        😡 Angry

        😨 Fearful

        🤢 Disgust

        😲 Surprised
        """)

    st.markdown("---")

    st.subheader("Technology Stack")

    st.write("• Python")
    st.write("• TensorFlow / Keras")
    st.write("• Scikit-Learn")
    st.write("• Librosa")
    st.write("• Plotly")
    st.write("• Streamlit")

    st.markdown("---")

    st.info("Upload a WAV audio file to begin emotion detection.")

# -------------------------------------------------------
# MAIN PAGE
# -------------------------------------------------------

st.title("🎤 EmotionSense AI Pro")

st.markdown("""
## AI Powered Speech Emotion Recognition using Machine Learning and Deep Learning

Detect emotions from speech using Machine Learning.

Upload a WAV file and receive:

- 🎯 Predicted Emotion
- 📊 Probability Distribution
- 📋 Prediction Table
- 🌊 Waveform
- 🎼 Spectrogram
- 🎹 MFCC Heatmap

---
""")

# -------------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------------

uploaded_file = st.file_uploader(
    "📂 Upload WAV File",
    type=["wav"]
)
st.write("Upload a WAV (.wav) audio file to analyze the speaker's emotion.")

# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

if uploaded_file is not None:
        
    audio_bytes = uploaded_file.getvalue()

    st.audio(audio_bytes)

    # ---------------------------------------------------
    # SAVE AUDIO TEMPORARILY
    # ---------------------------------------------------

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_path = tmp.name

    # ---------------------------------------------------
    # LOAD AUDIO
    # ---------------------------------------------------

    y, sr = librosa.load(audio_path, sr=None)

    duration = librosa.get_duration(y=y, sr=sr)

    rms = np.mean(
    librosa.feature.rms(y=y)
    )

    zcr = np.mean(
    librosa.feature.zero_crossing_rate(y)
    )

    # ---------------------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------------------

    start = time.time()

    with st.spinner("🎤 Analyzing speech..."):

        progress = st.progress(0)

        for i in range(100):
            progress.progress(i + 1)
            time.sleep(0.005)

        if selected_model == "Random Forest":
            emotion, probabilities = predict_emotion(audio_path)
        else:
            emotion, probabilities = predict_cnn(audio_path)


        end = time.time()

        prediction_time = end - start
    
        progress.empty()

    timestamp = datetime.now().strftime(
    "%d-%m-%Y %I:%M:%S %p"
    )

    st.success("✅ Prediction Completed!")

    if selected_model == "Random Forest":

        model_name = "Random Forest"

        algorithm = "Machine Learning"

        feature_vector = "156 Features"

        feature_type = "MFCC + Chroma + Mel"

        feature_caption = (
            "13 MFCC + 12 Chroma + 128 Mel Spectrogram + "
            "Zero Crossing Rate + RMS Energy + Spectral Centroid"
        )

    else:

        model_name = "CNN"

        algorithm = "Deep Learning"

        feature_vector = "40 × 130 MFCC"

        feature_type = "MFCC Spectrogram"

        feature_caption = (
            "40 MFCC coefficients preserved over time "
            "for CNN learning."
        )

    st.info(f"🤖 Current Prediction Model: **{model_name}**")

    # ---------------------------------------------------
    # PROBABILITY CHART
    # ---------------------------------------------------

    emotion_names = [
    "Angry",
    "Calm",
    "Disgust",
    "Fearful",
    "Happy",
    "Neutral",
    "Sad",
    "Surprised"
    ]

    prob_df = pd.DataFrame({
    "Emotion": emotion_names,
    "Probability": probabilities
    })

    # ---------------------------------------------------
    # PREDICTION RESULT
    # ---------------------------------------------------

    st.markdown("---")
    st.markdown("# 🎯 Prediction Result")

    confidence = float(probabilities.max() * 100)

    timestamp = datetime.now().strftime(
        "%d-%m-%Y %I:%M:%S %p"
    )

    emotion_icons = {
        "Happy": "😊",
        "Neutral": "😐",
        "Calm": "😌",
        "Sad": "😢",
        "Angry": "😡",
        "Fearful": "😨",
        "Disgust": "🤢",
        "Surprised": "😲"
    }

    # ---------------------------------------
    # SAVE PREDICTION HISTORY
    # ---------------------------------------

    history_item = {
        "Time": timestamp,
        "Emotion": f"{emotion_icons[emotion]} {emotion}",
        "Confidence": f"{confidence:.2f}%",
        "Model": model_name,
        "Prediction Time": f"{prediction_time:.2f} sec"
    }

    # Prevent duplicate entry on Streamlit reruns
    if (
        len(st.session_state.history) == 0 or
        st.session_state.history[-1] != history_item
    ):
        st.session_state.history.append(history_item)

    border_colors = {
        "Happy": "#22c55e",
        "Calm": "#10b981",
        "Neutral": "#9ca3af",
        "Sad": "#3b82f6",
        "Angry": "#ef4444",
        "Fearful": "#f97316",
        "Disgust": "#9333ea",
        "Surprised": "#eab308"
    }

    if confidence >= 80:
        badge = "🟢 HIGH CONFIDENCE"

    elif confidence >= 60:
        badge = "🟡 MEDIUM CONFIDENCE"

    else:
        badge = "🔴 LOW CONFIDENCE"


    left, right = st.columns([3.2,1.25], gap="medium")


    # ======================================================
    # LEFT CARD
    # ======================================================

    with left:

        st.markdown(
            f"""
    <div style="
    background:#1f2937;
    border-left:10px solid {border_colors[emotion]};
    border-radius:18px;
    padding:35px;
    height:470px;
    box-shadow:0 8px 25px rgba(0,0,0,.35);
    ">

    <h1 style="
    font-size:60px;
    margin-bottom:20px;
    color:{border_colors[emotion]};
    ">
    {emotion_icons[emotion]} {emotion}
    </h1>

    <h2 style="color:white;">
    Predicted Emotion
    </h2>

    <br>

    <h1 style="
    font-size:48px;
    color:#e5e7eb;
    ">
    {confidence:.2f}%
    </h1>

    <p style="
    font-size:22px;
    color:#bfc4cf;
    margin-top:-12px;
    ">
    Prediction Confidence
    </p>

    <hr>

    <p style="
    color:#bfc4cf;
    font-size:19px;
    ">

    <b>Model</b> :
    {model_name}

    <br><br>

    <b>Dataset</b> :
    RAVDESS

    </p>

    </div>
    """,
            unsafe_allow_html=True,
        )

        st.success(badge)


    # ======================================================
    # RIGHT CARD
    # ======================================================

    with right:

        with st.container(border=True):

            st.markdown("## 📊 Model Confidence")

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.progress(confidence / 100)

            st.metric(
                "Prediction Time",
                f"{prediction_time:.2f} sec"
            )

            st.metric(
                "Prediction Model",
                model_name
            )

            st.metric(
                "Dataset",
                "RAVDESS"
            )

            st.caption(timestamp)

    # ---------------------------------------------------
    # AI INTERPRETATION
    # ---------------------------------------------------

    emotion_description = {

        "Happy":
        "The speech contains energetic vocal patterns, expressive pitch variation, and positive voice dynamics. These characteristics are commonly associated with happiness.",

        "Calm":
        "The speaker's voice exhibits stable energy, balanced pitch, and smooth speech transitions, indicating a calm emotional state.",

        "Neutral":
        "The speech maintains consistent vocal characteristics without strong emotional cues, suggesting a neutral emotional state.",

        "Sad":
        "Lower vocal energy and softer speech dynamics indicate emotional characteristics commonly associated with sadness.",

        "Angry":
        "Higher vocal intensity, increased energy, and abrupt changes in speech suggest anger.",

        "Fearful":
        "The voice demonstrates irregular pitch variation and unstable vocal patterns that are often associated with fear.",

        "Disgust":
        "The vocal characteristics match acoustic patterns commonly observed in expressions of disgust.",

        "Surprised":
        "Rapid fluctuations in pitch and vocal intensity suggest a surprised emotional state."

    }

    st.subheader("🧠 AI Interpretation")

    if emotion in ["Happy", "Calm"]:
        st.success(emotion_description[emotion])

    elif emotion in ["Sad", "Neutral"]:
        st.info(emotion_description[emotion])

    elif emotion == "Angry":
        st.error(emotion_description[emotion])

    else:
        st.markdown(f"""
        <div style="
        background:#FFF8DC;
        padding:18px;
        border-radius:10px;
        border-left:6px solid orange;
        font-size:17px;
        ">
        🧠 {emotion_description[emotion]}
        </div>
        """, unsafe_allow_html=True)

    st.info(f"""
    ### 📌 Summary
    
    ✔ Predicted Emotion : **{emotion}**
    
    ✔ Confidence : **{confidence:.2f}%**
    
    ✔ Model : **{model_name}**
    
    ✔ Prediction Time : **{prediction_time:.2f} sec**

    """)

    st.subheader("ℹ️ Model Information")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Classification",
        model_name
    )

    m2.metric(
        "Algorithm",
        algorithm
    )

    m3.metric(
        "Dataset",
        "RAVDESS"
    )

    m4.metric(
        "Classes",
        "8"
    )

    m5, m6 = st.columns(2)

    m5.metric(
        "Feature Vector",
        feature_vector
    )

    m6.metric(
        "Task",
        "Speech Emotion Recognition"
    )

    st.caption(feature_caption)

    st.subheader("🎵 Audio Information")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "⏱ Duration",
        f"{duration:.2f} sec"
    )

    c2.metric(
        "🎧 Sample Rate",
        f"{sr:,} Hz"
    )

    c3.metric(
        "📈 RMS",
        f"{rms:.4f}"
    )

    c4.metric(
        "〰 Zero Crossing",
        f"{zcr:.4f}"
    )

    peak = np.max(np.abs(y))
    mean_amp = np.mean(np.abs(y))

    st.subheader("📈 Additional Audio Statistics")

    s1, s2 = st.columns(2)

    with s1:
        st.metric("Peak Amplitude", f"{peak:.4f}")

    with s2:
        st.metric("Average Amplitude", f"{mean_amp:.4f}")

    st.subheader("📈 Audio Visualization")
    
    left, right = st.columns(2)
    
    # -------------------------
    # Waveform
    # -------------------------

    with left:

        st.markdown("### 🌊 Waveform")

        fig, ax = plt.subplots(figsize=(9, 4.5))

        librosa.display.waveshow(
            y,
            sr=sr,
            ax=ax
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")

        st.pyplot(fig)

        plt.close(fig)


    # -------------------------
    # Spectrogram
    # -------------------------

    with right:

        st.markdown("### 🎼 Spectrogram")

        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(y)),
            ref=np.max
        )

        fig, ax = plt.subplots(figsize=(9, 4.5))

        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis="time",
            y_axis="hz",
            cmap="viridis",
            ax=ax
        )

        fig.colorbar(img, ax=ax)

        st.pyplot(fig)

        plt.close(fig)

    # ---------------------------------------------------
    # MFCC
    # ---------------------------------------------------

    st.subheader("🎹 MFCC Heatmap")

    mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=40
        )

    fig, ax = plt.subplots(figsize=(12, 4))

    img = librosa.display.specshow(
            mfcc,
            x_axis="time",
            cmap="coolwarm",
            ax=ax
        )

    fig.colorbar(img, ax=ax)

    st.pyplot(fig)

    plt.close(fig)

    st.divider()

    # ---------------------------------------
    # TOP 3 PREDICTIONS
    # ---------------------------------------

    st.subheader("🏅 Top 3 Predictions")

    top3 = (
        prob_df.sort_values(
            "Probability",
            ascending=False
        )
        .head(3)
        .reset_index(drop=True)
    )

    medals = ["🥇", "🥈", "🥉"]

    for i in range(3):

        col1, col2 = st.columns([5,1])

        with col1:
            emo = top3.iloc[i]["Emotion"]

            st.write(f"### {medals[i]} {emotion_icons[emo]} {emo}")

        with col2:
            st.metric(
                "Confidence",
                f"{top3.iloc[i]['Probability']*100:.2f}%"
            )

        st.progress(
            float(top3.iloc[i]["Probability"])
        )

    st.divider()

    st.subheader("📊 Prediction Probabilities")

    chart_df = prob_df.copy()

    chart_df["Probability"] = chart_df["Probability"] * 100

    chart_df["Color"] = chart_df["Emotion"].apply(
        lambda x: "Predicted" if x == emotion else "Other"
    )

    fig = px.bar(
        chart_df,
        x="Emotion",
        y="Probability",
        color="Color",
        text="Probability",
        color_discrete_map={
            "Predicted":"green",
            "Other":"lightgray"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Emotion",
        yaxis_title="Confidence (%)",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------------------------------
    # TABLE
    # ---------------------------------------------------

    st.subheader("📋 Probability Table")

    table = prob_df.copy()

    table = (
        table
        .sort_values("Probability", ascending=False)
        .head(5)
    )

    table["Emotion"] = table["Emotion"].map(
        lambda x: f"{emotion_icons[x]} {x}"
    )

    # Convert to percentage (still numeric)
    table["Probability"] = table["Probability"] * 100

    def highlight_prediction(row):
        if emotion in row["Emotion"]:
            return [
                "background-color:#d1fae5;font-weight:bold;"
            ] * len(row)
        return [""] * len(row)


    styled_table = (
        table.style
        .apply(highlight_prediction, axis=1)
        .background_gradient(
            subset=["Probability"],
            cmap="BuGn"
        )
        .format({
            "Probability":"{:.2f}%"
        })
    )

    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True
    )

    # ---------------------------------------------------
    # DOWNLOAD REPORT
    # ---------------------------------------------------

    current_time = datetime.now().strftime("%d-%m-%Y %I:%M:%S %p")

    prediction_report = f"""
    EmotionSense AI Prediction Report
    ==============================================

    Prediction Date & Time
    ----------------------
    {current_time}

    Prediction Time
    ---------------
    {prediction_time:.2f} sec

    Prediction Model
    ----------------
    {model_name}

    Predicted Emotion
    -----------------
    {emotion_icons[emotion]} {emotion}

    Confidence
    ----------
    {confidence:.2f}%

    Audio Information
    -----------------
    Duration           : {duration:.2f} sec
    Sample Rate        : {sr} Hz
    RMS Energy         : {rms:.6f}
    Zero Crossing Rate : {zcr:.6f}

    Top 3 Predictions
    -----------------
    """

    for i in range(3):
        prediction_report += (
            f"{i+1}. "
            f"{emotion_icons[top3.iloc[i]['Emotion']]} "
            f"{top3.iloc[i]['Emotion']} "
            f"- {top3.iloc[i]['Probability']*100:.2f}%\n"
        )

    prediction_report += f"""

    AI Interpretation
    -----------------
    {emotion_description[emotion]}

    ----------------------------------------------
    Generated by EmotionSense AI Pro
    Machine Learning Internship Project

    Developed by
    Jakkula Jayanthi
    ----------------------------------------------
    """

    st.download_button(
        "📄 Download Complete Prediction Report",
        prediction_report,
        file_name="EmotionSense_Report.txt",
        mime="text/plain",
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("🕘 Prediction History")

    if st.session_state.history:

        history_df = pd.DataFrame(st.session_state.history[::-1])

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("No predictions yet.")

    st.markdown("---")

    st.markdown(
    """
    ---
    <div style="text-align:center;padding:30px;">

    <h2>🎤 EmotionSense AI Pro</h2>

    <h4>Speech Emotion Recognition System</h4>

    <p>
    Developed using
    Python • Streamlit • TensorFlow • Scikit-Learn • Librosa • Plotly
    </p>

    <p>
    Dataset : <b>RAVDESS Emotional Speech Dataset</b>
    </p>

    <p>
    Developed by
    <b>Jakkula Jayanthi</b>
    </p>

    <p>
    © 2026 EmotionSense AI Pro
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )