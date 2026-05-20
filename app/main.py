from fastapi import FastAPI, HTTPException
from .rag_engine import get_answer_with_sources
from .models import QueryRequest, QueryResponse,Source



app = FastAPI(title="RAG Document QA API")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """
    Ask a question based on the loaded documents.
    Returns answer, relevant source chunks, and confidence level.
    """
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        result = get_answer_with_sources(request.question, k=request.k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

