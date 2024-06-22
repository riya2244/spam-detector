#app

import streamlit as st
import joblib

# Load the trained pipeline
pipeline = joblib.load('spam_classifier_pipeline.pkl')

# Preprocess function
def preprocess_text(text):
    processed_text = text.lower()
    return processed_text

# Streamlit app
st.set_page_config(page_title="Spam Message Detector", page_icon=":mailbox_with_mail:", layout="wide")

st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Spam Message Detector</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4B8BFF;'>Enter your message below to check if it's spam or not</h3>", unsafe_allow_html=True)

with st.container():
    user_input = st.text_area("Enter your message", height=200)

    if st.button('Detect'):
        preprocessed_text = preprocess_text(user_input)
        prediction = pipeline.predict([preprocessed_text])
        result = "Spam" if prediction == 1 else "Not Spam"
        st.markdown(f"<h2 style='text-align: center; color: {'#FF4B4B' if prediction == 1 else '#4B8BFF'};'>Prediction: {result}</h2>", unsafe_allow_html=True)

