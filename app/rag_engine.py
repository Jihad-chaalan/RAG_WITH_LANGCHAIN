# RAG LOgic , where i use the existing modules from RAG directory
from pathlib import Path
from RAG.documents_ingestion.documents_embeddings_and_storing import get_vector_store
from RAG.retrieval_and_generation.call_llm import get_llm
from RAG.retrieval_and_generation.rag_chain import create_rag_chain
from RAG.retrieval_and_generation.prompt_template import prompt


# ---------- Load existing vector store (no re‑embedding) ----------
print("Loading existing vector store...")
vector_store = get_vector_store(chunks=None)   # loads from ./chroma_db
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# ---------- Setup LLM ----------
llm = get_llm(temperature=0)


rag_chain = create_rag_chain(retriever, prompt, llm)

def get_answer_with_sources(question: str, k: int = 4):
    """
    Get answer, source chunks with similarity scores, and confidence.
    Reuses the existing vector_store and llm, but similarity scores require direct store call.
    """
    # 1. Retrieve docs with scores using the vector store directly
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=k)
    
    sources = []
    scores = []
    for doc, score in docs_with_scores:
        # Convert distance to similarity (assuming cosine distance with normalized embeddings)
        # Usually Chroma returns distance: 0 = identical, 2 = opposite; map to [0,1]
        similarity = 1 - min(score / 2, 1.0)
        scores.append(similarity)
        sources.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": round(similarity, 3)
        })
    
    # 2. Generate answer using the same context (reuse prompt and llm)
    context = "\n\n".join([doc.page_content for doc, _ in docs_with_scores])
    formatted_prompt = prompt.invoke({"context": context, "question": question})
    answer = llm.invoke(formatted_prompt).content
    
    # 3. Compute confidence based on highest similarity
    max_score = max(scores) if scores else 0.0
    if max_score >= 0.8:
        confidence = "high"
    elif max_score >= 0.5:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }