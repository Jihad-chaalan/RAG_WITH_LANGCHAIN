# Create Prompt Template
from langchain_core.prompts import ChatPromptTemplate

template = """You are a helpful assistant. Answer the question based on the context below.
If you cannot answer based on the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
