import os
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
from pinecone import ServerlessSpec, Pinecone
# from pinecone.grpc import PineconeGRPC as Pinecone


load_dotenv()  # Load environment variables from .env file

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = "audioindex"  # Choose a suitable index name

# Initialize Pinecone
pc=Pinecone(
    api_key=PINECONE_API_KEY
    # environment=PINECONE_ENVIRONMENT 
)

# Create Pinecone index with ServerlessSpec for cost-efficiency
spec = ServerlessSpec(
    cloud="aws",  # Choose your preferred cloud provider (aws, gcp, azure)
    region="us-east-1"  # Choose your desired region
)
existing_index_list = [index["name"] for index in pc.list_indexes()]
print("GR", existing_index_list)
print("VENKY")

# Create the index if it doesn't exist
if INDEX_NAME not in existing_index_list:
    pc.create_index(
        name=INDEX_NAME, 
        dimension=1536,  # Dimension for OpenAI's text-embedding-ada-002 model
        metric="cosine", 
        spec=spec
    )
else:
    print("Already data exits ")
# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") 

def process_audio_folder(folder_path, speaker_gender):
    """
    Processes audio files in a folder, generates embeddings, and stores them in Pinecone.

    Args:
        folder_path (str): Path to the folder containing audio files.
        speaker_gender (str): Gender of the speaker.
    """
    audio_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".wav")] 
    
    # Create a list of documents with audio file paths and speaker gender
    docs = [
        {"text": f"Audio file: {file_path}", "metadata": {"speaker_gender": speaker_gender}} 
        for file_path in audio_files
    ]

    # Create PineconeVectorStore
    vectorstore = PineconeVectorStore.from_documents(
        docs, 
        embedding=embeddings, 
        index_name=INDEX_NAME
    )

# Example Usage
male_folder = "sample_voice_data-master/males"
female_folder = "sample_voice_data-master/females"


process_audio_folder(male_folder, "male")
process_audio_folder(female_folder, "female")