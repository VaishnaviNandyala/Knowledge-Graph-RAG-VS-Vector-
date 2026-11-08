"""System S1 retrieval: classic vector RAG (embed chunks, return the ones most similar to the question)."""
from typing import List

import numpy as np

from .chunker import Chunk
from .types import RetrievalResult


class VectorRetriever:
    def __init__(self, chunks: List[Chunk], embedder, top_k: int = 4, budget_words: int = 120):
        self.chunks, self.embedder = chunks, embedder
        self.top_k, self.budget = top_k, budget_words
        self.matrix = embedder.encode([c.embed_text for c in chunks]).astype("float32")
        try:                                           # FAISS is optional; numpy is fine for small corpora
            import faiss
            self.index = faiss.IndexFlatIP(self.matrix.shape[1])
            self.index.add(self.matrix)
        except Exception:
            self.index = None

    def _search(self, qvec: np.ndarray, k: int):
        if self.index is not None:
            scores, ids = self.index.search(qvec.reshape(1, -1).astype("float32"), k)
            return list(zip(ids[0].tolist(), scores[0].tolist()))
        sims = self.matrix @ qvec
        ids = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in ids]

    def retrieve(self, question: str, budget_words: int = None) -> RetrievalResult:
        budget = self.budget if budget_words is None else budget_words
        qvec = self.embedder.encode([question])[0]
        ranked = self._search(qvec, min(len(self.chunks), self.top_k * 3))
        parts, items, titles, used = [], [], set(), 0
        for idx, score in ranked:
            if len(items) >= self.top_k:
                break
            chunk = self.chunks[idx]
            words = len(chunk.prompt_text.split())
            if used + words > budget:             # same word budget as every other system
                break
            parts.append(chunk.prompt_text)
            items.append({"title": chunk.title, "score": round(score, 3), "text": " ".join(chunk.sentences)})
            titles.add(chunk.title)
            used += words
        return RetrievalResult(text="\n".join(parts), titles=titles, items=items,
                               debug={"words_used": used, "ranked_top": [(self.chunks[i].title, round(s, 3)) for i, s in ranked[:6]]})
