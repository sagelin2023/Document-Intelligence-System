#indexing.py - builds a FAISS index for document chunks

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss 

from .embeddings import embed_texts #import the embedding function from embeddings.py

INDEX_DIR = Path("data/index")

# statistics about the index build process, @dataclass just means that it's a data container
@dataclass
class IndexBuildStats:
    doc_id: str #document identifier
    total_chunks_loaded: int
    total_chunks_indexed: int
    embedding_dim: int #dimension of the embedding vectors(number of items in each vector)

#takes the document id and loads the chunks from chunks.json
def _load_chunks(doc_id: str) -> List[Dict[str, Any]]:
    doc_dir = INDEX_DIR / doc_id
    chunks_path = doc_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found for doc_id={doc_id}")

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f) #takes the json file and turns it into a python object

    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a JSON list.")
    return chunks

#selects the texts and metadata from the chunks for embedding and indexing
def _select_texts_and_meta(
    chunks: List[Dict[str, Any]],
    min_chars: int = 30,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    
    texts: List[str] = [] #the actual text chunks to be embedded
    meta: List[Dict[str, Any]] = [] #references to the chunks

    for c in chunks:
        text = (c.get("text") or "")
        if len(text.strip()) < min_chars: #skip chunks that are too short
            continue

        texts.append(text) #add the text to the list of texts to embed
        #add metadata for this chunk
        meta.append(
            {
                "chunk_uid": c.get("chunk_uid"),
                "chunk_id": c.get("chunk_id"),
                "page_number": c.get("page_number"),
            }
        )

    return texts, meta

#builds the FAISS index for a document given its id
def build_index_for_doc(doc_id: str, write_meta: bool = True) -> IndexBuildStats:

    doc_dir = INDEX_DIR / doc_id
    if not doc_dir.exists():
        raise FileNotFoundError(f"doc directory not found: {doc_dir}")

    chunks = _load_chunks(doc_id)
    texts, meta = _select_texts_and_meta(chunks)

    # Creates the embeddings for the selected texts
    vectors = embed_texts(texts, normalize=True)
    if vectors.size == 0:
        raise ValueError("No valid chunks to index (all were empty/too short).")

    if vectors.ndim != 2:
        raise ValueError(f"Expected embeddings 2D array, got shape={vectors.shape}")

    n, dim = vectors.shape #n is number of vectors, dim is dimension of each vector

    # Build exact search index (flat, inner product)
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Save index
    index_path = doc_dir / "index.faiss"
    faiss.write_index(index, str(index_path))

    # Save meta mapping (recommended)
    if write_meta:
        meta_path = doc_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    return IndexBuildStats(
        doc_id=doc_id,
        total_chunks_loaded=len(chunks),
        total_chunks_indexed=index.ntotal,
        embedding_dim=dim,
    )
