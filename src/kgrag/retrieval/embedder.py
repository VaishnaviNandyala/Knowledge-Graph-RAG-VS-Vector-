"""Turn text into vectors so we can measure similarity.

Two interchangeable backends with the same interface:
    fit(list_of_texts)   -> learn from the corpus (TF-IDF needs it, SBERT does not)
    encode(list_of_texts)-> numpy array (n, dim), each row L2-normalised, so dot product = cosine similarity
"""
import math
import re
from collections import Counter
from typing import List

import numpy as np

STOPWORDS = set("a an the of in on at to is was were are be been by for with and or as it its this that "
                "which who whom where when what from into than then there their his her he she they".split())


def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


class TfidfEmbedder:
    """Classical bag-of-words TF-IDF, written in numpy so the toy demo needs no downloads.
    (Syllabus link: your 'basic text processing' unit.) It matches WORDS, not meaning:
    'born' will not match 'birth'. That limitation is part of what the demo shows."""

    def __init__(self):
        self.vocab, self.idf = {}, None

    def fit(self, texts: List[str]) -> "TfidfEmbedder":
        df = Counter()
        for t in texts:
            df.update(set(tokenize(t)))
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in sorted(df)])
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        mat = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            for w, c in Counter(tokenize(t)).items():
                j = self.vocab.get(w)
                if j is not None:
                    mat[i, j] = (1 + math.log(c)) * self.idf[j]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        return mat / np.maximum(norms, 1e-9)


class SbertEmbedder:
    """Dense neural embeddings with Sentence-BERT (Reimers & Gurevych, 2019).
    It matches MEANING, so 'born' and 'birthplace' end up close.
    Setup: pip install sentence-transformers   (downloads ~90 MB for MiniLM)"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def fit(self, texts: List[str]) -> "SbertEmbedder":
        return self                      # pretrained: nothing to learn

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
