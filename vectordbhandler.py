import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer
from groq import Groq
# Load environment variables
load_dotenv()

# API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "audioagentindex"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it doesn’t exist
existing_indexes = [index["name"] for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Adjust based on the embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index {INDEX_NAME} already exists.")

# Initialize Hugging Face Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def groq_client(api_key):
    return Groq(api_key=api_key,)


# Function to get embeddings
def get_embedding(text):
    return embedding_model.encode(text).tolist()

# Function to convert non-WAV files to WAV
def convert_to_wav(file_path):
    if not file_path.endswith(".wav"):
        audio = AudioSegment.from_file(file_path)
        wav_path = file_path.rsplit(".", 1)[0] + ".wav"
        audio.export(wav_path, format="wav")
        return wav_path
    return file_path

# Function to transcribe an audio file (Replace with Whisper if needed)
def transcribe_audio_file_dummy(audio_file):
    # This function should be replaced with your Whisper or ASR model
    return "Sample transcription for " + audio_file  # Dummy transcription
def transcribe_audio_file(client, audio_file):

    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(model="distil-whisper-large-v3-en",file=(audio_file, f.read()),)
        return transcript.text

# Function to process and store audio embeddings
def process_audio_folder(folder_path, speaker_gender):
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".wav")]
    docs = []

    for file_path in audio_files:
        file_path = convert_to_wav(file_path)  # Ensure it's in WAV format
        client = groq_client(os.getenv("GROQ_API_KEY"))
        transcription = transcribe_audio_file(client, file_path)
        print("Transcription: ", transcription)

        embedding = get_embedding(transcription)  # Generate embeddings

        metadata = {"speaker_gender": speaker_gender, "transcription": transcription}
        docs.append({"id": file_path, "values": embedding, "metadata": metadata})

    # Insert embeddings into Pinecone
    if docs:
        index = pc.Index(INDEX_NAME)
        index.upsert(vectors=docs)
        print(f"Stored embeddings for {len(docs)} audio files in Pinecone.")

# Define folder paths
male_folder = "sample_voice_data-master/males"
female_folder = "sample_voice_data-master/females"

# Process and store embeddings
process_audio_folder(male_folder, "male")
process_audio_folder(female_folder, "female")

# import os
# import whisper
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import OpenAIEmbeddings
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Load Whisper model for transcription
# whisper_model = whisper.load_model("base")

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "audioindex"

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Create Pinecone index with ServerlessSpec
# spec = ServerlessSpec(
#     cloud="aws",  
#     region="us-east-1"
# )

# # Check for existing index
# existing_index_list = [index["name"] for index in pc.list_indexes()]
# if INDEX_NAME not in existing_index_list:
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=1536,  # Dimension for OpenAI's text-embedding-ada-002 model
#         metric="cosine", 
#         spec=spec
#     )
# else:
#     print(f"Index {INDEX_NAME} already exists.")

# # Initialize embeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# def extract_audio_embeddings(file_path):
#     """
#     Extracts embeddings from the given audio file using Whisper.

#     Args:
#         file_path (str): Path to the audio file.
    
#     Returns:
#         embedding (str): Transcription or textual representation of the audio.
#     """
#     # Use Whisper model to transcribe the audio file
#     result = whisper_model.transcribe(file_path)
#     transcription = result["text"]
    
#     # Generate embeddings for the transcribed text
#     embedding = embeddings.embed_text(transcription)
#     return embedding, transcription

# def process_audio_folder(folder_path, speaker_gender):
#     """
#     Processes audio files in a folder, generates embeddings, and stores them in Pinecone.

#     Args:
#         folder_path (str): Path to the folder containing audio files.
#         speaker_gender (str): Gender of the speaker.
#     """
#     audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".wav")]
    
#     # Process each audio file and store embeddings in Pinecone
#     docs = []
#     for file_path in audio_files:
#         embedding, transcription = extract_audio_embeddings(file_path)
        
#         # Create metadata for the audio file
#         metadata = {"speaker_gender": speaker_gender, "transcription": transcription}
        
#         # Add the document to the list
#         docs.append({"text": transcription, "embedding": embedding, "metadata": metadata})
    
#     # Store the documents in Pinecone
#     vectorstore = PineconeVectorStore.from_documents(
#         docs,
#         embedding=embeddings,
#         index_name=INDEX_NAME
#     )

#     print(f"Stored embeddings for {len(docs)} audio files in Pinecone.")

# # Example Usage
# male_folder = "sample_voice_data-master/males"
# female_folder = "sample_voice_data-master/females"

# process_audio_folder(male_folder, "male")
# process_audio_folder(female_folder, "female")
