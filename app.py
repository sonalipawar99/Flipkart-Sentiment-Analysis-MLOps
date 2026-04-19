import streamlit as st
import joblib
from utils import clean_text

model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("Flipkart Review Sentiment Analysis")
user_input = st.text_area("Enter your review:")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    
    if prediction[0] == 1:
        st.success("Positive Sentiment! ")
    else:
        st.error("Negative Sentiment! ")