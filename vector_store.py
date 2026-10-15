"""vector_store.py - Embeddings + FAISS index (Steps 5, 6 and retrieval)."""
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder():
    return SentenceTransformer(EMBED_MODEL)


class VectorStore:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.chunks = []  # same order as the vectors in the index

    def build(self, chunks, progress_callback=None):
        """Embed all chunks (in batches, so we can show progress) and add to FAISS."""
        texts = [c["text"] for c in chunks]
        vectors = []
        batch = 64
        for i in range(0, len(texts), batch):
            v = self.embedder.encode(
                texts[i:i + batch], normalize_embeddings=True, convert_to_numpy=True
            )
            vectors.append(v)
            if progress_callback:
                progress_callback(min(i + batch, len(texts)) / len(texts))
        matrix = np.vstack(vectors).astype("float32")
        # vectors are normalized, so inner product == cosine similarity
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        self.chunks = chunks

    def search(self, query, k=3):
        """Return a list of (chunk, score) for the k closest chunks."""
        q = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, ids = self.index.search(q.astype("float32"), min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]
