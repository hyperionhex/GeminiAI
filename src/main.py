import streamlit as st
from chat import run_chat_model
from vision import run_vision_model
from pdf import run_pdf_model
from yt import run_yt_model

def main():
    st.title("Google Gemini-Pro AI Models")
    models = ["Chat", "Vision", "PDF", "YouTube"]
    selected_model = st.selectbox("Select a model", models)

    if selected_model == "Chat":
        run_chat_model()
    elif selected_model == "Vision":
        run_vision_model()
    elif selected_model == "PDF":
        run_pdf_model()
    elif selected_model == "YouTube":
        run_yt_model()

if __name__ == "__main__":
    main()
