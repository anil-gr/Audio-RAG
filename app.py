import streamlit as st
from audio_recorder_streamlit import audio_recorder
from groq import Groq
import os

st.set_page_config(page_title = "AUDIO RAG", page_icon = "🎙️", layout = "centered")

def groq_client(api_key):
    return Groq(api_key=api_key,)

def transcribe_audio_file(client, audio_file):

    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(model="distil-whisper-large-v3-en",file=(audio_file, f.read()),)
        return transcript.text

def llm_response(client, prompt):
    completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role":"user","content": prompt}],)

    return completion.choices[0].message.content

def main():
    st.sidebar.title("API KEY CONFIGURATION")
    api_key = st.sidebar.text_input("Enter Your Groq API KEY", type="password")

    st.title("🎤 AUDIO Chatbot 💬")
    st.html("<h4>Hey, Hi. Click on the Voice recorder below to interact with me. How can i assist you today?</h4>")
    if api_key:
        client = groq_client(api_key)
        recorded_audio = audio_recorder()
        if recorded_audio:
            audio_file = "audio.mp3"
            with open(audio_file, "wb") as f:
                f.write(recorded_audio)
            prompt = transcribe_audio_file(client, audio_file)
            st.write("Transcribed Text : ", prompt)
            response = llm_response(client, prompt)
            st.write("LLM Response : ",response)

            # Delete the audio file after it is used
            if os.path.exists(audio_file):
                os.remove(audio_file)

if __name__=="__main__":
    main()