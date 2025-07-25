import streamlit as st
import joblib
import os

# Load the trained model and vectorizer
model_path = os.path.join("model", "spam_classifier.pkl")
vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Streamlit UI
st.set_page_config(page_title="Email Spam Detector", layout="centered")
st.title("📧 Email Spam Detector")
st.write("This app uses a trained machine learning model to classify emails as SPAM or HAM (not spam).")

# Input
email_text = st.text_area("✉️ Enter Email Content", height=200)

# Prediction
if st.button("Predict"):
    if not email_text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        transformed = vectorizer.transform([email_text])
        prediction = model.predict(transformed)[0]
        if prediction == "spam":
            st.error("🚫 This is SPAM")
        else:
            st.success("✅ This is HAM (Not Spam)")
