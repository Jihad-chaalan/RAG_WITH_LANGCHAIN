# 📚 RAG Document QA System

A production‑ready **Retrieval-Augmented Generation (RAG)** pipeline that answers questions from your documents (PDF, CSV, TXT, JSON).  
Uses free local embeddings (`sentence-transformers`) and Groq’s fast LLM API.

## ✨ Features

- Load multiple document types: PDF, CSV, TXT, JSON
- Recursive file scanning (supports subfolders)
- Smart text chunking with configurable size/overlap
- Local embeddings with `all-MiniLM-L6-v2` (free, private, no API key)
- Vector store with Chroma (persisted to disk)
- LLM integration with Groq (fast inference, cheap)
- LCEL (LangChain Expression Language) for clean, composable chains
- Interactive Q&A loop with source‑aware answers

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- [Groq API key](https://console.groq.com) (free tier)
- (Optional) [LangSmith API key](https://smith.langchain.com) for tracing

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/rag-document-qa.git
cd rag-document-qa
Create virtual environment

bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
Install dependencies

bash
pip install -r requirements.txt
Set up environment variables

Copy .env.example to .env and fill in your keys:

bash
cp .env.example .env
# Edit .env with your actual API keys
Prepare your documents

Place your PDF, CSV, TXT, or JSON files inside RAG/data_sample/ (create the folder if needed).

Usage
Build the vector store (one time)
From the project root:

bash
python RAG/main.py
This will load, chunk, embed, and store your documents in Chroma (location defined by CHROMA_PERSIST_DIR).

Run the FastAPI server
bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
Then send POST requests to http://localhost:8000/ask with JSON body:

json
{
  "question": "What is pattern recognition?",
  "k": 4
}
Example response:

json
{
  "answer": "Pattern recognition is the field that...",
  "sources": [
    {
      "content": "text snippet...",
      "metadata": {"source": "book.pdf", "page": 42},
      "similarity_score": 0.92
    }
  ],
  "confidence": "high"
}
text

---
```
