import streamlit as st
import joblib
import re

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# Streamlit UI
st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review here")

if st.button("Analyze Sentiment"):
    cleaned = clean_text(review)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    st.markdown(f"### Sentiment: {'✅ Positive' if prediction == 1 else '❌ Negative'}")
