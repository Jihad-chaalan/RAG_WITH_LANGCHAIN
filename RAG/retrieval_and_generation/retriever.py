#step of retriever the relevant chunks with the query from chroma
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever(vector_store, search_type="similarity", k=4):
    """
    Create a retriever from a Chroma vector store.
    """
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    print("✅ Retriever created")
    return retriever