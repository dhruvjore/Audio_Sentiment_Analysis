import streamlit as st
from pydub import AudioSegment
import tempfile
import os
import google.generativeai as genai
from dotenv import load_dotenv
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any

load_dotenv()

# Configure Google API for audio processing
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe the audio using Google's Generative AI."""
    model = genai.GenerativeModel("models/gemini-1.5-flash-001")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Transcribe this dialogue in the format of timecode, speaker, caption. Use speaker A, speaker B, etc. to identify speakers. If the speaker continues, just continue with the timestamp and do not show different timestamps. Output it in a pretty dialogue format.",
            audio_file
        ]
    )
    return response.text

def summarize_audio(audio_file_path: str) -> str:
    """Summarize the audio using Google's Generative AI."""
    model = genai.GenerativeModel("models/gemini-1.5-flash-001")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Summarize the audio using Google's Generative API.",
            audio_file
        ]
    )
    return response.text

def analyze_entities_and_sentiment(audio_file_path: str) -> str:
    """Analyze entities and sentiment in the audio using Google's Generative AI."""
    model = genai.GenerativeModel("models/gemini-1.5-flash-001")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Please identify the main entities mentioned in this audio and provide a sentiment analysis with polarity scores for each entity. Format the output as a list of entities with their associated sentiment and polarity score. The polarity score must not be a range and it should be a precise value. Since this is a healthcare domain audio we need to focus more on parts where the healthcare representative talks and the way care was given and also focus on when the customer talks about the healthcare reviews to give a proper overall sentiment of the care provided",
            audio_file
        ]
    )
    return response.text

def identify_speakers(audio_file_path: str) -> str:
    """Identify speakers in the audio using Google's Generative AI."""
    model = genai.GenerativeModel("models/gemini-1.5-flash-001")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(
        [
            "Please identify the speakers in the following audio.",
            audio_file
        ]
    )
    return response.text

def adaptive_sentiment_trend_analyzer(audio_file_path: str, time_window: int, sentiment_threshold: float) -> Dict[str, Any]:
    """Analyze sentiment trends with adaptive time windows and custom thresholds."""
    model = genai.GenerativeModel("models/gemini-1.5-flash-001")
    audio_file = genai.upload_file(path=audio_file_path)
    
    prompt = f"""
    Analyze the sentiment in this audio file using the following parameters:
    1. Time window: {time_window} seconds
    2. Sentiment threshold for significant changes: {sentiment_threshold}

    For each time window:
    1. Provide a sentiment score (-1 to 1)
    2. Identify key phrases or words driving the sentiment
    3. Note any significant sentiment shifts (exceeding the threshold)

    Also, provide:
    1. An overall sentiment trend analysis
    2. Recommendations for addressing negative sentiments or capitalizing on positive ones
    3. Identification of potential trigger points for sentiment changes

    Format your response using the following Python function. Replace the placeholder values with your analysis:

    def get_sentiment_analysis():
        return {{
            "sentiment_data": [
                {{"timestamp": "00:00", "score": 0.0, "key_phrases": ["phrase1", "phrase2"], "significant_shift": False}},
                # Add more entries as needed
            ],
            "overall_trend": "Describe the overall trend here",
            "recommendations": [
                "Recommendation 1",
                "Recommendation 2",
                # Add more recommendations as needed
            ],
            "trigger_points": [
                {{"timestamp": "00:00", "description": "Describe the trigger point"}},
                # Add more trigger points as needed
            ],
            "heatmap_data": [
                # Add heatmap data here in the format:
                # {{"time": "00:00", "score": 0.0}}
            ]
        }}

    Only include the Python function in your response. Please include nothing else except the function do not start with ```python just start with der.
    """
    
    response = model.generate_content([prompt, audio_file])
    print(response)
    
    # Extract the Python function from the response
    function_str = response.text.strip()
    if function_str.startswith('```python'):
        # Remove the prefix '```python' and any potential leading/trailing whitespace
        function_str = function_str[len('```python'):].strip()
        function_str = function_str.strip('```').strip()
    
    # Create a local namespace to execute the function
    local_namespace = {}
    exec(function_str, local_namespace)
    
    # Call the function and get the result
    result = local_namespace['get_sentiment_analysis']()
    
    return result

def plot_sentiment_trend(sentiment_data: List[Dict[str, Any]]):
    """Create an interactive plot of sentiment trends."""
    fig = go.Figure()
    timestamps = [data['timestamp'] for data in sentiment_data]
    scores = [data['score'] for data in sentiment_data]
    fig.add_trace(go.Scatter(x=timestamps, y=scores, mode='lines+markers', name='Sentiment Score'))
    fig.update_layout(title='Sentiment Trend Over Time',
                      xaxis_title='Timestamp',
                      yaxis_title='Sentiment Score')
    return fig

def plot_sentiment_heatmap(heatmap_data: List[Dict[str, Any]]):
    """Create a heatmap of sentiment data."""
    if not heatmap_data:
        return None
    
    # Convert the heatmap data to a DataFrame
    df = pd.DataFrame(heatmap_data)
    
    # Ensure 'time' column is in datetime format without specifying the format
    df['time'] = pd.to_datetime(df['time'], errors='coerce')  # Use errors='coerce' to handle invalid parsing
    
    # Check if conversion was successful
    if df['time'].isnull().all():
        raise ValueError("All time data could not be parsed. Please check the time format.")
    
    # Prepare data for heatmap
    df.set_index('time', inplace=True)
    df_resampled = df.resample('T').mean().fillna(0)  # Resample by minute and fill missing values

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=df_resampled.values.T,
        x=df_resampled.index,
        y=['Sentiment'],
        colorscale='Viridis'
    ))
    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Sentiment',
        title='Sentiment Heatmap'
    )
    
    return fig

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Streamlit app interface
st.set_page_config(page_title="Audio Analysis App", layout="wide")

st.title('🎙️ Audio Analysis App')

with st.expander("ℹ️ About this app"):
    st.write("""
        This advanced audio analysis app utilizes Google's Generative AI to provide comprehensive insights into your audio files:
        
        1. 📝 **Transcription**: Get a detailed transcript of the dialogue with speaker identification and timestamps.
        2. 📊 **Summarization**: Receive a concise summary of the main points discussed in the audio.
        3. 🧠 **Entity and Sentiment Analysis**: Identify key entities mentioned and their associated sentiment with polarity scores.
        4. 👥 **Speaker Identification**: Recognize distinct speakers in the audio content.
        5. 📈 **Adaptive Sentiment Trend Analysis**: Analyze sentiment trends over time with customizable parameters.
        
        Upload your audio file in WAV or MP3 format to unlock these powerful analysis features!
    """)

audio_file = st.file_uploader("📤 Upload Audio File", type=['wav', 'mp3'])

if audio_file is not None:
    audio_path = save_uploaded_file(audio_file)
    st.audio(audio_path)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('🎭 Transcribe Audio', use_container_width=True):
            with st.spinner('Transcribing...'):
                transcription = transcribe_audio(audio_path)
                st.subheader('📜 Transcription')
                st.text_area("", transcription, height=300)
        
        if st.button('📝 Summarize Audio', use_container_width=True):
            with st.spinner('Summarizing...'):
                summary = summarize_audio(audio_path)
                st.subheader('📋 Summary')
                st.info(summary)
    
    with col2:
        if st.button('🔍 Analyze Entities & Sentiment', use_container_width=True):
            with st.spinner('Analyzing...'):
                analysis = analyze_entities_and_sentiment(audio_path)
                st.subheader('🧐 Entity & Sentiment Analysis')
                st.write(analysis)
        
        if st.button('👥 Identify Speakers', use_container_width=True):
            with st.spinner('Identifying speakers...'):
                speaker_info = identify_speakers(audio_path)
                st.subheader('🎤 Speaker Identification Results')
                st.write(speaker_info)

    st.subheader("📊 Adaptive Sentiment Trend Analyzer")
    col3, col4 = st.columns(2)
    with col3:
        time_window = st.slider("Time Window (seconds)", 10, 60, 30, 5)
    with col4:
        sentiment_threshold = st.slider("Sentiment Change Threshold", 0.1, 0.5, 0.2, 0.05)

    if st.button('Analyze Sentiment Trends', use_container_width=True):
        with st.spinner('Analyzing sentiment trends...'):
            sentiment_data = adaptive_sentiment_trend_analyzer(audio_path, time_window, sentiment_threshold)
            
            st.subheader("Sentiment Trend Analysis")
            st.plotly_chart(plot_sentiment_trend(sentiment_data['sentiment_data']))
            
            st.subheader("Key Insights")
            st.write("Overall Trend:", sentiment_data['overall_trend'])
            
            st.subheader("Recommendations")
            for rec in sentiment_data['recommendations']:
                st.write(f"- {rec}")

            st.subheader("Trigger Points")
            for trigger in sentiment_data['trigger_points']:
                st.write(f"- At {trigger['timestamp']}: {trigger['description']}")

    if st.button('📥 Export Detailed Report'):
        # Placeholder for report generation and download functionality
        st.write("Report export functionality would be implemented here.")

st.markdown("---")
st.markdown("Powered by Google's Generative AI 🚀")
