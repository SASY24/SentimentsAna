import streamlit as st
from transformers import pipeline
import emoji

# Load the sentiment analysis model
model_name = "poom-sci/WangchanBERTa-finetuned-sentiment"
sentiment_analyzer = pipeline('sentiment-analysis', model=model_name)

# Streamlit app
st.title("Thai Sentiment Analyzer 🤖")

# Input text
text_input = st.text_area("Enter Thai text for sentiment analysis", "ขอความเห็นหน่อย...")

# Button to trigger analysis
if st.button("Analyze Sentiment"):
    # Analyze sentiment using the model
    results = sentiment_analyzer([text_input])

    # Extract sentiment and score
    sentiment = results[0]['label']
    score = results[0]['score']

    # Display result with visual enhancements
    st.subheader("Sentiment Analysis Result:")
    if sentiment == 'pos':
        st.success(f"**Positive Sentiment** 😊 (Score: {score:.2f})")
        st.progress(score)
    elif sentiment == 'neg':
        st.error(f"**Negative Sentiment** 🙁 (Score: {score:.2f})")
        st.progress(score)
    else:
        st.warning(f"**Neutral Sentiment** 🤔 (Score: {score:.2f})")
        st.progress(score)

    # Display original text and sentiment explanation
    st.write("**Original Text:**")
    st.write(text_input)

    if sentiment == 'pos':
        st.write("This text expresses positive emotions, such as happiness, joy, or approval.")
    elif sentiment == 'neg':
        st.write("This text expresses negative emotions, such as sadness, anger, or disapproval.")
    else:
        st.write("This text expresses neutral emotions, without strong positive or negative feelings.")

    # Add an interactive element: A simple quiz
    st.subheader("Test Your Sentiment Analysis Skills!")
    quiz_text = "ลองวิเคราะห์ความรู้สึกของข้อความนี้: 'วันนี้เหนื่อยมากเลย'"
    quiz_answer = "Negative"
    user_answer = st.text_input("Your answer:")
    if st.button("Submit"):
        if user_answer.lower() == quiz_answer.lower():
            st.success("Correct! You're a sentiment analysis pro!")
        else:
            st.error(f"Incorrect. The correct answer is: {quiz_answer}")
