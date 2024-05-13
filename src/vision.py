from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

# Set page configuration (this should be the first Streamlit command)
st.set_page_config(page_title="Gemini Image Demo")

# Load environment variables
load_dotenv()

# Configure Google Generative AI API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Function to get response from Gemini-Pro Vision model
def get_gemini_response(input, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input != "":
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

def run_vision_model():
    st.header("Gemini Application")

    # Input prompt
    input = st.text_input("Input Prompt: ", key="input")

    # Image uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Submit button
    submit = st.button("Tell me about the image")

    # Generate response from Gemini-Pro Vision model
    if submit:
        response = get_gemini_response(input, image)
        st.subheader("The Response is")
        st.write(response)
