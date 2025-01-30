import streamlit as st
from audio_recorder_streamlit import audio_recorder
from groq import Groq
import pyttsx3
import os
from pinecone import Pinecone
from langchain_community.tools.youtube.search import YouTubeSearchTool
from sentence_transformers import SentenceTransformer


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "audioagentindex"

pc = Pinecone(api_key=PINECONE_API_KEY)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dimensional embeddings


class VoiceInputAgent:
    def __init__(self, api_key):
        self.api_key = api_key
        self.client = Groq(api_key=api_key)

    def record_audio(self):
        """Record audio through the Streamlit component and return audio bytes."""
        return audio_recorder()

    def transcribe_audio(self, audio_bytes):
        """Transcribe the recorded audio into text."""
        audio_file = "audio.mp3"
        with open(audio_file, "wb") as f:
            f.write(audio_bytes)

        with open(audio_file, "rb") as f:
            transcript = self.client.audio.transcriptions.create(
                model="distil-whisper-large-v3-en", file=f
            )

        os.remove(audio_file)  
        return transcript.text


class SearchAgent:
    def __init__(self, youtube_api_key):
        self.youtube_tool = YouTubeSearchTool(api_key=youtube_api_key)
        self.pinecone_index = pc.Index(INDEX_NAME)  

    def get_embedding(self, text):
        """Generate an embedding for the given text."""
        return embedding_model.encode(text).tolist()

    def search_pinecone(self, query):
        """Search Pinecone for the closest matching audio transcript."""
        query_embedding = self.get_embedding(query)

        response = self.pinecone_index.query(
            vector=query_embedding,
            top_k=1,
            include_metadata=True
        )

        if response['matches']:
            return {'source': 'pinecone', 'result': response['matches'][0]['metadata']['transcription'], 'path': response['matches'][0]['id']}
        return None

    def search_youtube(self, query):
        """Search YouTube for a matching video."""
        return self.youtube_tool.run(query)

    def search(self, query):
        """Search both Pinecone and YouTube for a result."""
        pinecone_result = self.search_pinecone(query)
        return pinecone_result if pinecone_result else {'source': 'youtube', 'result': self.search_youtube(query)}


class ResponseAgent:
    def __init__(self):
        self.engine = pyttsx3.init()

    def play_audio(self, audio_path):
        """Play the retrieved audio file."""
        if os.path.exists(audio_path):
            st.audio(audio_path)
        else:
            st.write("Audio file not found!")

    def generate_text_response(self, youtube_url):
        """Generate a text response with a YouTube link."""
        response = f"Here is the link to the YouTube video: {youtube_url}"
        st.write(response)
        self.engine.runAndWait()


def main():
    st.sidebar.title("API KEY CONFIGURATION")
    youtube_api_key = os.getenv("YOUTUBE_API_KEY")

    voice_input_agent = VoiceInputAgent(api_key=os.getenv("GROQ_API_KEY"))
    search_agent = SearchAgent(youtube_api_key=youtube_api_key)
    response_agent = ResponseAgent()

    st.title("🎤 Voice Search Bot 💬")
    st.write("**Click on the Voice recorder below to interact with me.**")

    recorded_audio = voice_input_agent.record_audio()

    if recorded_audio:
        text_input = voice_input_agent.transcribe_audio(recorded_audio)
        st.write("**Transcribed Text:**", text_input)

        search_results = search_agent.search(text_input)

        if search_results['source'] == 'pinecone':
            response_agent.play_audio(search_results['path'])
            st.write("Text of this audio file", search_results['result'])
        elif search_results['source'] == 'youtube':
            response_agent.generate_text_response(search_results['result'])
        else:
            st.write("No matching results found in Pinecone or YouTube.")


if __name__ == "__main__":
    main()
    