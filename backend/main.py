from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import pdfplumber, uuid, json

app = FastAPI()
UPLOAD_DIR = Path("data/uploads") 
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)#ensure the upload directory exists
INDEX_DIR = Path("data/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)#ensure the index directory exists
CHUNK_SIZE = 1200
OVERLAP = 200

def chunk_text(text, chunk_size, overlap):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    
    chunks = []
    start = 0
    stride = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += stride
    
    return chunks
    

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    #if the file has no name
    if not file.filename:
        raise HTTPException(status_code=400, detail ="Missing filename")
    #if the file type is not a pdf
    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {file.content_type}")
    
    #unique identifier for the document
    doc_id = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{doc_id}.pdf" #path to save the file

    #writes the file to disk in chunks
    bytes_saved = 0 #tracks how many bytes have been saved
    with open(save_path, "wb") as out: #opens the file at the path in write-binary mode and name it out
        #reads the file in chunks until there is no more data
        while True:
            chunk = await file.read(1024 * 1024)  # Read in 1 MB chunks
            if not chunk: #if no more data is left
                break
            out.write(chunk) #writes the chunk to the file
            bytes_saved += len(chunk) #updates the byte counter
    

    page_texts = [] #list to hold the text from each page
    chunks = [] #list to hold the text chunks
    chunk_id = 0 #unique identifier for each chunk
    with pdfplumber.open(save_path) as pdf: #opens the saved pdf file
        pages = len(pdf.pages) #number of pages in the pdf
        page_number = 1
        for page in pdf.pages:
            text = page.extract_text()
            if not text: #if there is no text on the page
                text = ""
            page_texts.append(text) #adds the text to the list
            for chunk in chunk_text(text, CHUNK_SIZE, OVERLAP): #for each chunk on the page
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "page_number": page_number,
                    "text": chunk
                })
                chunk_id += 1
            page_number += 1 #increment the page number 
    full_text = "\n".join(page_texts) #joins all the text parts into a single string
    chars_extracted = len(full_text)
    preview = full_text[:500]  # First 500 characters as preview


    doc_index_dir = INDEX_DIR / doc_id #directory to save the index for this document
    doc_index_dir.mkdir(parents=True, exist_ok=True) #ensure the directory exists
    chunks_path = doc_index_dir / "chunks.json" #path to save the chunks
    with open(chunks_path, "w", encoding="utf-8") as f: #opens the file in write mode
        json.dump(chunks, f, ensure_ascii=False, indent=2) #saves the chunks as json

    total_chunks = len(chunks)
    avg_len = (sum(len(c["text"]) for c in chunks) / total_chunks) if total_chunks else 0

    #printing chunk summary to console
    print("doc_id:", doc_id)
    print("total_chunks:", total_chunks)
    print("avg_chunk_length:", avg_len)
    if total_chunks:
        sample = chunks[0].copy() #take a sample chunk
        sample["text"] = sample["text"][:200]  # shorten for console
        print("sample_chunk:", sample)

    return {
        "doc_id": doc_id,
        "pages": pages,
        "chars_extracted": chars_extracted,
        "preview": preview,
        "total_chunks": len(chunks)
    }

@app.post("/search")
async def search_document():
    return {
        "query": "...",
        "k": 5,
        "doc_id": "...:"
    }
