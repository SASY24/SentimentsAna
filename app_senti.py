

import streamlit as st
from transformers import pipeline

# Load the sentiment analysis model
model_name = "poom-sci/WangchanBERTa-finetuned-sentiment"
sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)

# Streamlit app
st.title("Thai Sentiment Analysis App")

# Input text
text_input = st.text_area("Enter Thai text for sentiment analysis", "ขอความเห็นหน่อย... ")

# Button to trigger analysis
if st.button("Analyze Sentiment"):
    # Analyze sentiment using the model
    results = sentiment_analyzer([text_input])

    # Extract sentiment and score
    sentiment = results[0]['label']
    score = results[0]['score']
    

    # Display result as progress bars
    st.subheader("Sentiment Analysis Result:")

    if sentiment == 'pos':
        st.success(f"Positive Sentiment (Score: {score:.2f})")
        st.progress(score)
    elif sentiment == 'neg':
        st.error(f"Negative Sentiment (Score: {score:.2f})")
        st.progress(score)
    else:
        st.warning(f"Neutral Sentiment (Score: {score:.2f})")
        st.progress(score)
