# AI Document Intelligence System

A simple Retrieval-Augmented Generation (RAG) system that allows users to upload PDFs, index them, and ask questions that are answered strictly using the document content, with citations to the source text. 

---

## Tech Stack

- Backend: FastAPI  
- Embeddings: sentence-transformers  
- Vector Search: FAISS  
- LLM: Google Gemini 
- Language: Python  

---

## How to Use

1. Start the FastAPI server:
   ```bash
   python -m uvicorn backend.main:app --reload
2. Open the API Documentation
3. Upload a PDF using POST /upload and save the doc_id
4. Index the document with POST /index/doc_id
5. Ask questions using POST/ask with the doc_id and the question.