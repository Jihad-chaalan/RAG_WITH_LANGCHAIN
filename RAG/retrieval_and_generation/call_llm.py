#create LLM

# retrieval_and_generation/llm_setup.py
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_llm(temperature=0, model_name="openai/gpt-oss-120b"):
    """
    Initialize and return a Groq LLM instance.
    
    """
    llm = ChatGroq(
        temperature=temperature,
        model=model_name,
        groq_api_key=os.getenv("GROQ_API_KEY")  
    )

    print("✅ LLM initialized with Groq")
    
    return llm





