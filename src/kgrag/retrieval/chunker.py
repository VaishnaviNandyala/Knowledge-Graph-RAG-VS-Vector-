"""Split documents into chunks for the vector index."""
from dataclasses import dataclass
from typing import List

from ..data import Doc


@dataclass
class Chunk:
    title: str
    sentences: List[str]

    @property
    def embed_text(self) -> str:             # what we embed: title + sentences (the title helps a lot)
        return self.title + ". " + " ".join(self.sentences)

    @property
    def prompt_text(self) -> str:            # what the LLM sees: one sentence per line under a title header
        return "### " + self.title + "\n" + "\n".join(self.sentences)


def chunk_docs(docs: List[Doc], max_words: int = 60) -> List[Chunk]:
    chunks = []
    for doc in docs:
        cur, n = [], 0
        for s in doc.sentences:
            w = len(s.split())
            if cur and n + w > max_words:
                chunks.append(Chunk(doc.title, cur))
                cur, n = [], 0
            cur.append(s)
            n += w
        if cur:
            chunks.append(Chunk(doc.title, cur))
    return chunks
