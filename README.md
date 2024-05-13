## Overview
This project integrates Google's Gemini-Pro AI models into a Streamlit web application to provide a suite of AI-powered tools. Each tool leverages a specific aspect of the Gemini-Pro model to interact with different types of data, including chat, vision, PDFs, and YouTube videos. The application is designed to showcase the capabilities of AI in processing and generating content based on user inputs.

## Features
1. **Chat Model ([chat.py](/src/chat.py))**:
   - Implements a chatbot using the Gemini-Pro model.
   - Users can interact with the chatbot, which processes and responds to queries in real-time.

2. **Vision Model ([vision.py](/src/vision.py))**:
   - Utilizes the Gemini-Pro vision model to analyze images.
   - Users can upload images and receive descriptions or insights based on the visual content.

3. **PDF Model ([pdf.py](/src/pdf.py))**:
   - Processes uploaded PDF documents to extract text.
   - Performs text chunking and vectorization for efficient search and retrieval.
   - Supports a question-answering feature where users can ask questions related to the content of the uploaded PDFs.

4. **YouTube Model ([yt.py](/src/yt.py))**:
   - Extracts transcripts from YouTube videos using the YouTube Transcript API.
   - Summarizes the video content using the Gemini-Pro text generation model, providing concise notes on the video's content.

## Technologies
- **Python**: Primary programming language used.
- **Streamlit**: Framework for building the web interface.
- **Google Generative AI**: API used for accessing Gemini-Pro models.
- **YouTube Transcript API**: For retrieving video transcripts.
- **PyPDF2 and LangChain**: For PDF processing and text analysis.

## Setup and Installation
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Set up environment variables for the Google API key in a `.env` file.
4. Run the application using `streamlit run main.py`.

## Usage
Select the desired model from the dropdown menu in the application:
- Chat: Interact with the AI chatbot.
- Vision: Upload images to analyze.
- PDF: Upload PDFs for text extraction and querying.
- YouTube: Enter a YouTube video link to get detailed notes.
