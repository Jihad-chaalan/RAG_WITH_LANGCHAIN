#Step of storing documents after embedding
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_vector_store(chunks, persist_directory="./chroma_db", model_name="all-MiniLM-L6-v2"):
    """
    Create or load a Chroma vector store from document chunks using free sentence transformers embeddings.

    """
    
    # 1. Initialize the embedding model 
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
    )
    print(f"✅ Model {model_name} ready (dimension: 384 for MiniLM)")

    # 2. Check if vector store already exists
    if Path(persist_directory).exists():
        print(f"Loading existing vector store from {persist_directory}...")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print("✅ Existing vector store loaded")
    else:
        print(f"Creating new vector store at {persist_directory} (this may take a few minutes)...")
        # This call embeds all chunks and builds the Chroma index
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"✅ Vector store created with {len(chunks)} chunks")
    
    return vector_store