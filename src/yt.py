import streamlit as st
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Define the prompt for the Gemini-Pro model
prompt = """You are a YouTube video summarizer. Summarize the transcript text
of the entire video, highlighting key points in a concise format within 250 words.
Summary of the provided text:"""

# Function to retrieve transcript data from a YouTube video
from youtube_transcript_api import TranscriptsDisabled

def get_transcript(youtube_link, language_code='en'):
    if "watch?v=" in youtube_link:
        video_id = youtube_link.split("watch?v=")[1]
    elif "youtu.be/" in youtube_link:
        video_id = youtube_link.split("youtu.be/")[1]
    else:
        return "Invalid YouTube link"
    
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=[language_code])
        transcript = " ".join(segment["text"] for segment in transcript_data)
        return transcript
    except TranscriptsDisabled:
        st.error(f"Subtitles are disabled for the video {youtube_link}. Please try another video.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Function to generate summary using the Gemini-Pro model
def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

def run_yt_model():
    st.title("YouTube Transcript to Detailed Notes Converter")

    # User input for YouTube video link
    youtube_link = st.text_input("Enter YouTube Video Link:")

    if youtube_link:
        video_id = youtube_link.split("=")[1] if "watch?v=" in youtube_link else youtube_link.split("youtu.be/")[1]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

        # Button to generate detailed notes
        if st.button("Get Detailed Notes"):
            transcript_text = get_transcript(youtube_link)
            if transcript_text:
                summary = generate_gemini_content(transcript_text, prompt)
                st.markdown("## Detailed Notes:")
                st.write(summary)
