import streamlit as st
from transformers import pipeline
import speech_recognition as sr
from streamlit_lottie import st_lottie
import requests

# Page config
st.set_page_config(page_title="🧠 Sentiment Analyzer", layout="wide")

# Background gradient and floating emojis using CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%);
        background-attachment: fixed;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .floating-emoji {
        position: fixed;
        font-size: 3rem;
        animation: floaty 6s ease-in-out infinite;
        opacity: 0.7;
    }
    .emoji1 { top: 20%; left: 10%; animation-delay: 0s;}
    .emoji2 { top: 50%; left: 80%; animation-delay: 2s;}
    .emoji3 { top: 70%; left: 40%; animation-delay: 4s;}
    @keyframes floaty {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    </style>

    <div class="floating-emoji emoji1">😊</div>
    <div class="floating-emoji emoji2">🎤</div>
    <div class="floating-emoji emoji3">🧠</div>
    """,
    unsafe_allow_html=True,
)

# Function to load Lottie animations from URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Sidebar with Lottie animation and instructions
with st.sidebar:
    st.header("🔍 How to Use")
    st.markdown("""
    - Type or **speak** your input.
    - Click **Analyze Emotion**.
    - View the most accurate emotion highlighted.
    - See top 3 emotions with confidence bars & emojis.
    - Powered by 🤗 Hugging Face model.
    """)
    lottie_microphone = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_znlcqdo2.json")
    if lottie_microphone:
        st_lottie(lottie_microphone, height=150, key="mic")

# Page title and subtitle
st.markdown(
    "<h1 style='text-align: center; font-weight: bold;'>🧠 AI-Based Sentiment Analyzer with Voice 🎤</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Type or speak your feelings and see how an AI understands your emotions!</p>",
    unsafe_allow_html=True,
)

# Load model (cached)
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification", 
        model="joeddav/distilbert-base-uncased-go-emotions-student", 
        return_all_scores=True
    )

model = load_model()

# Voice input function
def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, could not understand your speech.")
        except sr.RequestError:
            st.error("Speech recognition service is unavailable.")
    return ""

# Initialize session state variable to store user input persistently
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# UI input areas
col1, col2 = st.columns([4, 1])

with col1:
    st.session_state.user_input = st.text_area(
        "📝 Type your text here:", 
        value=st.session_state.user_input, 
        height=150
    )

with col2:
    if st.button("🎤 Use Voice Input"):
        voice_text = get_voice_input()
        if voice_text:
            st.session_state.user_input = voice_text  # update session state

# Analyze button
if st.button("🔍 Analyze Emotion"):
    if not st.session_state.user_input.strip():
        st.warning("Please enter or speak some text.")
    else:
        with st.spinner("Analyzing..."):
            results = model(st.session_state.user_input)[0]
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            best_emotion = results[0]
            other_emotions = results[1:3]

            emoji_map = {
                "admiration": "👏",
                "amusement": "😄",
                "anger": "😠",
                "annoyance": "😒",
                "approval": "👍",
                "caring": "🤗",
                "confusion": "😕",
                "curiosity": "🤔",
                "desire": "😍",
                "disappointment": "😞",
                "disapproval": "👎",
                "disgust": "🤢",
                "embarrassment": "😳",
                "excitement": "🤩",
                "fear": "😨",
                "gratitude": "🙏",
                "grief": "😭",
                "joy": "😊",
                "love": "❤️",
                "nervousness": "😬",
                "optimism": "🌞",
                "pride": "🏆",
                "realization": "💡",
                "relief": "😌",
                "remorse": "😔",
                "sadness": "😢",
                "surprise": "😲",
                "neutral": "😐"
            }

            # Show the most accurate emotion prominently
            st.subheader("🌟 Most Accurate Emotion Detected:")
            label = best_emotion['label']
            score = round(best_emotion['score'] * 100, 2)
            emoji = emoji_map.get(label.lower(), "❓")
            st.markdown(
                f"""
                <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                    <div style='font-size: 2.5rem; margin-right: 20px;'>{emoji}</div>
                    <div>
                        <h2 style='margin: 0;'>{label.capitalize()}</h2>
                        <p style='font-size: 1rem; margin: 0;'>{score}% confidence</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Show other top emotions with progress bars
            st.subheader("🎯 Other Top Emotions:")
            for item in other_emotions:
                label = item['label']
                score = round(item['score'] * 100, 2)
                emoji = emoji_map.get(label.lower(), "❓")
                st.markdown(
                    f"""
                    <div style='display: flex; align-items: center; margin-bottom: 10px;'>
                        <div style='font-size: 2rem; margin-right: 10px;'>{emoji}</div>
                        <div style='flex-grow: 1;'>
                            <strong>{label.capitalize()}</strong> — {score}%
                            <div style='background: #eee; border-radius: 10px; overflow: hidden; margin-top: 5px;'>
                                <div style='width: {score}%; background: linear-gradient(90deg, #a6c1ee, #fbc2eb); height: 15px; border-radius: 10px;'></div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
