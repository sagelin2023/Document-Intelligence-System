#embeddings.py - turns the chunks into vectors to perform vector search
from __future__ import annotations # helps with forward references in type hints

import numpy as np #handles vector math
from sentence_transformers import SentenceTransformer #the model that creates the embeddings

_MODEL_NAME = "all-MiniLM-L6-v2" #type of model being used
_model: SentenceTransformer | None = None #initially none then becomes the loaded model

#ensures the model is loaded
def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model

#normalizes the vectors to all be the same length, only caring about the direction of each vector
#need normalization for cosine similarity search so search results are accurate
#mat is a numpy array where each row is a vector
def _l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) #calculates the length of each vector
    return mat / np.maximum(norms, eps) #divides each vector by its length to make it length 1

#creates the embeddings for a list of texts(our chunks)
def embed_texts(texts: list[str], normalize: bool = True) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32) #return empty matrix if no texts

    model = _get_model()
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False) #creates the vectors for each text

    emb = np.asarray(emb, dtype=np.float32)   # ensure float32 for FAISS

    if normalize:
        emb = _l2_normalize(emb) #normalizes the vectors

    return emb

#embeds the query string into a vector
def embed_query(query: str, normalize: bool = True) -> np.ndarray:
    query = (query or "").strip()
    if not query:
        raise ValueError("Query is empty.")
    return embed_texts([query], normalize=normalize)
