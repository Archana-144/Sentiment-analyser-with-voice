import streamlit as st
from transformers import pipeline
import speech_recognition as sr

# Set page config
st.set_page_config(page_title="Emotion Detector with Voice", layout="centered")

st.title("😊 Emotion Detector with Voice Input")

@st.cache_resource
def load_model():
    return pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

model = load_model()

# Voice input section
if st.button("Record Voice"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak now.")
        audio = recognizer.listen(source, phrase_time_limit=5)
    try:
        user_input = recognizer.recognize_google(audio)
        st.success(f"Recognized Text: {user_input}")
        
        # Emotion detection
        results = model(user_input)[0]
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        st.success("Detected Emotions:")
        for res in results[:3]:
            label = res['label']
            score = round(res['score'] * 100, 2)
            st.write(f"**{label}**: {score}%")
            st.progress(min(int(score), 100))

    except Exception as e:
        st.error(f"Could not understand audio: {e}")

# Also allow manual text input
user_input = st.text_area("Or type your text here:", height=150)
if st.button("Analyze Text"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        results = model(user_input)[0]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        st.success("Detected Emotions:")
        for res in results[:3]:
            label = res['label']
            score = round(res['score'] * 100, 2)
            st.write(f"**{label}**: {score}%")
            st.progress(min(int(score), 100))
