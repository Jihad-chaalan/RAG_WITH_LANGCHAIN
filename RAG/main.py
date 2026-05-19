from documents_ingestion.document_loader import load_documents
from documents_ingestion.documents_chunking import chunk_documents
from documents_ingestion.documents_embeddings_and_storing import get_vector_store
from retrieval_and_generation.rag_chain import create_rag_chain
from retrieval_and_generation.prompt_template import prompt
from retrieval_and_generation.call_llm import get_llm
import os


# test = "data_sample"
# result = load_documents(test)

# # Access the list of Document objects
# docs = result["documents"]

# print(f"✅ Total chunks loaded: {result['total_chunks']}")
# print(f"📁 Files loaded: {result['files_loaded']}")
# print(f"❌ Failed files: {result['failed_files']}\n")

# # Inspect the first 3 documents (or all if less)
# for i, doc in enumerate(docs[:5]):  # show first 5 chunks
#     print(f"\n{'='*60}")
#     print(f"📄 Document #{i+1}")
#     print(f"{'='*60}")
    
#     # METADATA
#     print("\n📋 METADATA:")
#     for key, value in doc.metadata.items():
#         print(f"   {key}: {value}")
    
#     # CONTENT PREVIEW (first 300 characters)
#     print("\n📝 CONTENT PREVIEW:")
#     preview = doc.page_content[:300].replace('\n', ' ')
#     print(f"   {preview}...")
#     print(f"   (Total length: {len(doc.page_content)} characters)")




# raw_docs = result["documents"]

# print(f"Loaded {len(raw_docs)} raw documents")

# # Step 2: Chunk
# chunks = chunk_documents(raw_docs, chunk_size=1000, chunk_overlap=200)

# print(f"Created {len(chunks)} chunks")

# # Inspect first chunk
# if chunks:
#     print(f"\nFirst chunk preview: {chunks[0].page_content[:200]}")
#     print(f"Metadata: {chunks[0].metadata}")


def main():
    # 1. Load
    print("Loading documents...")
    load_result = load_documents("./data_sample")
    if not load_result["documents"]:
        print("No documents loaded. Exiting.")
        return
    raw_docs = load_result["documents"]
    print(f"Loaded {len(raw_docs)} raw documents")

    # 2. Chunk
    print("Chunking documents...")
    chunks = chunk_documents(raw_docs)
    print(f"Created {len(chunks)} chunks")

    # 3. Embed & store (creates or loads vector store)
    print("Creating/loading vector store...")
    vector_store = get_vector_store(chunks)   # your existing function
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # 4. Setup LLM (Groq)
    llm = get_llm(temperature=0)


    rag_chain = create_rag_chain(retriever, prompt, llm)


    print("\n=== RAG System Ready ===")
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        if question.lower() in ("exit", "quit"):
            break
        answer = rag_chain.invoke(question)
        print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    main()