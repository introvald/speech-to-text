import streamlit as st
import speech_recognition as sr
from transformers import pipeline

# Set up the NLP model for sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

# Function to recognize speech
def recognize_speech():
    # Create a recognizer instance
    recognizer = sr.Recognizer()
    # Use the microphone as the source
    with sr.Microphone() as source:
        st.write("Please speak something...")
        audio_data = recognizer.listen(source)  # Listen for the input
        st.write("Recognizing...")
        try:
            text = recognizer.recognize_google(audio_data)  # Recognize the speech
            st.success("You said: " + text)
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service.")
            return None

# Streamlit app layout
st.title("Speech Recognition and NLP App")
st.write("This app converts your speech into text and analyzes the sentiment of the text.")

# Button to start speech recognition
if st.button("Start Speech Recognition"):
    recognized_text = recognize_speech()

    if recognized_text:  # If speech was recognized successfully
        st.subheader("Sentiment Analysis")
        sentiment_result = sentiment_analyzer(recognized_text)
        st.write("Sentiment:", sentiment_result[0]['label'])
        st.write("Confidence Score:", sentiment_result[0]['score'])

