import streamlit as st
from transformers import pipeline
import speech_recognition as sr
from PIL import Image
import base64

# Page config
st.set_page_config(page_title="Emotion Detector", layout="wide")

# 🌈 Background (gradient using CSS)
def add_bg_from_url():
    st.markdown("""
        <style>
        body {
            background: linear-gradient(to right, #fbc2eb, #a6c1ee);
            background-attachment: fixed;
        }
        </style>
        """, unsafe_allow_html=True)
add_bg_from_url()

# Sidebar
with st.sidebar:
    st.header("🔍 How to Use")
    st.markdown("""
    - Type or **speak** your input.
    - Click **Analyze Emotion**.
    - View top 3 emotions with confidence bars & emojis.
    - Powered by 🤗 Hugging Face model.
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ using Streamlit")

# Title
st.markdown("<h1 style='text-align: center;'>😊 Emotion Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Type or speak your feelings and see how an AI understands your emotions!</p>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="joeddav/distilbert-base-uncased-go-emotions-student", return_all_scores=True)

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

# Text input
user_input = st.text_area("📝 Type your text here:", height=150)
use_voice = st.button("🎤 Use Voice Input")

if use_voice:
    user_input = get_voice_input()

# Emotion detection
if st.button("🔍 Analyze Emotion"):
    if user_input.strip() == "":
        st.warning("Please enter or speak some text.")
    else:
        with st.spinner("Analyzing..."):
            results = model(user_input)[0]
            results = sorted(results, key=lambda x: x['score'], reverse=True)
            top_emotions = results[:3]

            st.subheader("🎯 Top Emotions Detected:")
            emoji_map = {
                "joy": "😊", "sadness": "😢", "anger": "😠", "fear": "😨",
                "surprise": "😲", "love": "❤️", "gratitude": "🙏", "optimism": "🌞",
                "pride": "🏆", "disapproval": "👎", "nervousness": "😬", "curiosity": "🤔"
            }

            for item in top_emotions:
                label = item['label']
                score = round(item['score'] * 100, 2)
                emoji = emoji_map.get(label.lower(), "")
                st.write(f"**{emoji} {label.capitalize()}**: {score}%")
                st.progress(min(int(score), 100))
