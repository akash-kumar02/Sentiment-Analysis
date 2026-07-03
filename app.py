import streamlit as st
import joblib

# Load model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Sentiment Analysis")

st.title("😊 Sentiment Analysis App")

st.write("Enter any review below.")

text = st.text_area("Review")

if st.button("Predict"):

    if text.strip() == "":
        st.warning("Please enter text")
    else:

        x = vectorizer.transform([text])

        prediction = model.predict(x)[0]

        if prediction == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")