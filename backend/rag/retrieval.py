# retrieval.py - performs vector search on the indexed documents
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss  # type: ignore

from .embeddings import embed_query

INDEX_DIR = Path("data/index")

# loads the FAISS index for a given document id
def _load_index(doc_id: str) -> faiss.Index:
    doc_dir = INDEX_DIR / doc_id
    index_path = doc_dir / "index.faiss"
    if not index_path.exists():
        raise FileNotFoundError(f"index.faiss not found for doc_id={doc_id}. Build the index first.")
    return faiss.read_index(str(index_path))

# loads the chunks for a given document id
def _load_chunks(doc_id: str) -> List[Dict[str, Any]]:
    doc_dir = INDEX_DIR / doc_id
    chunks_path = doc_dir / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.json not found for doc_id={doc_id}")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    if not isinstance(chunks, list):
        raise ValueError("chunks.json must contain a JSON list.")
    return chunks

# loads the meta information for a given document id, if it exists
def _load_meta_if_exists(doc_id: str) -> List[Dict[str, Any]] | None:
    doc_dir = INDEX_DIR / doc_id
    meta_path = doc_dir / "meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, list):
        raise ValueError("meta.json must contain a JSON list.")
    return meta

# builds a lookup dictionary from chunk_uid to chunk object
def _build_chunk_lookup(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        uid = c.get("chunk_uid")
        if uid:
            lookup[str(uid)] = c
    return lookup

#embeds the user query, searches FAISS for the top-k similar chnk vectors, and maps those FAISS results back to the original chunks
def search_doc(doc_id: str, query: str, k: int = 5) -> Dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be > 0")

    index = _load_index(doc_id) #loads the FAISS index
    chunks = _load_chunks(doc_id) #loads the chunks
    meta = _load_meta_if_exists(doc_id) #loads the meta if it exists

    qvec = embed_query(query, normalize=True)  # embeds the query (1, dim)

    # FAISS returns:
    # scores: shape (1, k), indices: shape (1, k)
    scores, ids = index.search(qvec, k)

    scores_list = scores[0].tolist() #flatten to 1D list
    ids_list = ids[0].tolist() #flatten to 1D list

    # Prepare mapping to full chunk objects
    chunk_lookup = _build_chunk_lookup(chunks)

    results: List[Dict[str, Any]] = []
    for score, row in zip(scores_list, ids_list):
        # FAISS may return -1 if not enough vectors (rare with Flat, but still possible)
        if row == -1:
            continue

        chunk_obj: Dict[str, Any] | None = None

        if meta is not None:
            # Use meta row -> chunk_uid
            m = meta[row]
            uid = m.get("chunk_uid")
            if uid is not None:
                chunk_obj = chunk_lookup.get(str(uid))
        else:
            # Fallback: assume FAISS rows align with chunks list order
            if 0 <= row < len(chunks):
                chunk_obj = chunks[row]

        if chunk_obj is None:
            # If mapping fails, still return the row+score for debugging
            results.append({"score": float(score), "row": int(row), "error": "Missing chunk mapping"})
            continue

        results.append(
            {
                "score": float(score),
                "chunk_uid": chunk_obj.get("chunk_uid"),
                "chunk_id": chunk_obj.get("chunk_id"),
                "page_number": chunk_obj.get("page_number"),
                "text": chunk_obj.get("text"),
            }
        )

    return {"doc_id": doc_id, "query": query, "k": k, "results": results}
