import streamlit as st
import requests

# --- Page Configuration ---
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="centered"
)

# --- App Title and Description ---
st.title("Real-Time Sentiment Analysis 🤖")
st.markdown("""
Enter a movie review, a product comment, or any text below to find out if the sentiment is positive or negative. 
This app sends your text to a machine learning model deployed on Google Cloud Run!
""")

# --- API Endpoint ---
# This is the URL of your deployed Google Cloud Run API
API_URL = "https://sentiment-api-service-105420428646.us-central1.run.app/predict"

# --- User Input ---
with st.form(key='sentiment_form'):
    user_text = st.text_area("Enter your text here:", height=150)
    submit_button = st.form_submit_button(label='Analyze Sentiment ✨')

# --- Handle Form Submission ---
if submit_button:
    if not user_text:
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # --- Call the API ---
                response = requests.post(API_URL, json={'text': user_text})
                response.raise_for_status()  # Raise an exception for bad status codes
                
                result = response.json()
                sentiment = result.get('sentiment')
                probability = result.get('probability')

                # --- Display the Result ---
                st.success("Analysis complete!")
                
                if sentiment == 'positive':
                    st.markdown(f"### 😊 Sentiment: **Positive**")
                else:
                    st.markdown(f"### 😞 Sentiment: **Negative**")
                
                st.metric(label="Confidence Score", value=f"{probability:.2%}")

                # Optional: Show a progress bar as a visual element
                st.progress(probability)

            except requests.exceptions.RequestException as e:
                st.error(f"Could not connect to the API. Please try again later. Error: {e}")

# --- Footer ---
st.markdown("---")
st.write("Powered by FastAPI, Google Cloud Run, and Streamlit.")
