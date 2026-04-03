"""
Semantic embedding using Sentence Transformers.
We chose this over simple fuzzy match because it handles synonyms, conceptual 
overlaps, and complex queries much better, fulfilling the "cosine sim" requirement
while behaving like a real production RAG system.
"""
import os
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

class Embedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        if HAS_SENTENCE_TRANSFORMERS:
            # We use all-MiniLM-L6-v2 because it provides a good balance between 
            # performance (small, 80MB) and semantic representation quality.
            print(f"[Embedder] Loading sentence-transformer model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            print("[Embedder] Model loaded successfully.")
        else:
            print("[Embedder] WARNING: sentence_transformers not installed. Using fallback random embedder.")
            print("[Embedder] Please run `pip install sentence-transformers` for real semantic similarity.")
            self.model = None

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Produce normalized dense vectors for the input texts.
        """
        if not texts:
            return np.array([])
            
        if self.model:
            embeddings = self.model.encode(texts)
            # Normalize embeddings to efficiently use np.dot for cosine similarity
            if len(embeddings.shape) == 1:
                embeddings = embeddings.reshape(1, -1)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.where(norms == 0, 1e-10, norms) # avoid division by zero
            return embeddings
        else:
            # Fallback for compilation testing
            embeddings = np.random.rand(len(texts), 384)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            return embeddings / norms

    @staticmethod
    def cosine_similarity(query_emb: np.ndarray, doc_emb: np.ndarray) -> np.ndarray:
        """
        Given a normalized 1D query embedding (or 2D 1xN) and a 2D array of normalized doc embeddings, 
        returns a 1D array of similarity scores.
        """
        # Ensure query is 1D
        query_emb = query_emb.flatten()
        return np.dot(doc_emb, query_emb)
