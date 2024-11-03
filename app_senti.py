import streamlit as st
import pandas as pd
from transformers import pipeline
import emoji
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json
import os

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load the sentiment analysis model
@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis', model="poom-sci/WangchanBERTa-finetuned-sentiment")

sentiment_analyzer = load_model()

# Function to save analysis history
def save_analysis(text, sentiment, score):
    analysis = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'text': text,
        'sentiment': sentiment,
        'score': score
    }
    st.session_state.history.append(analysis)

# Function to get emoji based on score
def get_sentiment_emoji(score, sentiment):
    if sentiment == 'pos':
        if score > 0.8:
            return "🤗"
        elif score > 0.6:
            return "😊"
        else:
            return "🙂"
    else:
        if score > 0.8:
            return "😢"
        elif score > 0.6:
            return "😕"
        else:
            return "🙁"

# Main app
st.title("Advanced Thai Sentiment Analyzer 🤖")
st.markdown("---")

# Sidebar for additional features
with st.sidebar:
    st.header("Features")
    show_history = st.checkbox("Show Analysis History")
    show_stats = st.checkbox("Show Statistics")
    
# Main content area with tabs
tab1, tab2, tab3 = st.tabs(["Single Analysis", "Batch Analysis", "Learning Zone"])

with tab1:
    # Single text analysis
    text_input = st.text_area("ป้อนข้อความภาษาไทยที่ต้องการวิเคราะห์", "ขอความเห็นหน่อย...", height=100)
    
    if st.button("วิเคราะห์ความรู้สึก", key="single_analysis"):
        with st.spinner("กำลังวิเคราะห์..."):
            results = sentiment_analyzer([text_input])
            sentiment = results[0]['label']
            score = results[0]['score']
            
            # Save to history
            save_analysis(text_input, sentiment, score)
            
            # Display result with enhanced visuals
            col1, col2 = st.columns(2)
            
            with col1:
                if sentiment == 'pos':
                    st.success(f"ความรู้สึกเชิงบวก {get_sentiment_emoji(score, sentiment)}")
                else:
                    st.error(f"ความรู้สึกเชิงลบ {get_sentiment_emoji(score, sentiment)}")
                    
            with col2:
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = score * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgray"},
                            {'range': [33, 66], 'color': "gray"},
                            {'range': [66, 100], 'color': "darkgray"}
                        ],
                    },
                    title = {'text': "ความมั่นใจในการวิเคราะห์"}
                ))
                st.plotly_chart(fig)

with tab2:
    # Batch analysis
    st.header("วิเคราะห์หลายข้อความพร้อมกัน")
    batch_text = st.text_area(
        "ป้อนข้อความหลายบรรทัด (แต่ละบรรทัดจะถูกวิเคราะห์แยกกัน)",
        "สวัสดีวันจันทร์\nวันนี้เหนื่อยจัง\nความสุขที่แสนวิเศษ",
        height=150
    )
    
    if st.button("วิเคราะห์ทั้งหมด"):
        texts = batch_text.split('\n')
        results = sentiment_analyzer(texts)
        
        # Create results DataFrame
        df = pd.DataFrame({
            'ข้อความ': texts,
            'ความรู้สึก': [r['label'] for r in results],
            'คะแนน': [r['score'] for r in results]
        })
        
        # Display results in a nice table
        st.dataframe(df.style.background_gradient(subset=['คะแนน']))
        
        # Show distribution chart
        fig = px.pie(df, names='ความรู้สึก', title='การกระจายของความรู้สึก')
        st.plotly_chart(fig)

with tab3:
    # Learning zone
    st.header("🎓 โซนการเรียนรู้")
    
    # Example sentences
    examples = {
        "ฉันมีความสุขมากวันนี้": "pos",
        "เหนื่อยจังเลย ไม่ไหวแล้ว": "neg",
        "วันนี้อากาศดีมาก": "pos",
        "เสียใจที่ทำงานไม่สำเร็จ": "neg"
    }
    
    st.subheader("✍️ ทดสอบความเข้าใจ")
    selected_example = st.selectbox("เลือกประโยคตัวอย่าง:", list(examples.keys()))
    
    user_guess = st.radio("ทายความรู้สึกของประโยคนี้:", ['pos', 'neg'])
    
    if st.button("ตรวจคำตอบ"):
        if user_guess == examples[selected_example]:
            st.success("🎉 ถูกต้อง! คุณเก่งมาก")
        else:
            st.error("❌ ไม่ถูกต้อง ลองใหม่อีกครั้ง")
        
        # Show model's analysis
        result = sentiment_analyzer([selected_example])[0]
        st.info(f"การวิเคราะห์จาก AI: {result['label']} (ความมั่นใจ: {result['score']:.2f})")

# Display history if enabled
if show_history:
    st.markdown("---")
    st.header("📊 ประวัติการวิเคราะห์")
    
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        st.dataframe(history_df)
        
        # Download button for history
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="⬇️ ดาวน์โหลดประวัติ (CSV)",
            data=csv,
            file_name="sentiment_history.csv",
            mime="text/csv"
        )
    else:
        st.info("ยังไม่มีประวัติการวิเคราะห์")

# Show statistics if enabled
if show_stats and st.session_state.history:
    st.markdown("---")
    st.header("📈 สถิติการวิเคราะห์")
    
    history_df = pd.DataFrame(st.session_state.history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment distribution over time
        fig = px.line(history_df, x='timestamp', y='score', color='sentiment',
                     title='แนวโน้มความรู้สึกตามเวลา')
        st.plotly_chart(fig)
    
    with col2:
        # Sentiment distribution pie chart
        fig = px.pie(history_df, names='sentiment',
                    title='สัดส่วนความรู้สึกทั้งหมด')
        st.plotly_chart(fig)
