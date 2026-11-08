"""The five systems compared in the paper. They differ ONLY in how the context is built.

    S0  no retrieval            lower bound
    S1  vector RAG              the baseline
    S2  KG-RAG                  the proposed method
    S3  hybrid                  graph facts + text chunks
    S4  oracle                  gold paragraphs handed to the LLM: upper bound for the generator
"""
from dataclasses import dataclass
from typing import Dict, List

from .generation.prompt import INSUFFICIENT, build_prompt
from .retrieval.types import RetrievalResult


@dataclass
class Output:
    answer: str
    retrieval: RetrievalResult


class _Base:
    name = "base"

    def __init__(self, llm):
        self.llm = llm

    def retrieve(self, q: dict) -> RetrievalResult:
        raise NotImplementedError

    def run(self, q: dict) -> Output:
        r = self.retrieve(q)
        return Output(self.llm.generate(build_prompt(r.text, q["question"])), r)


class NoRetrieval(_Base):
    name = "S0 LLM only"

    def retrieve(self, q):
        return RetrievalResult(text="")


class VectorRAG(_Base):
    name = "S1 Vector RAG"

    def __init__(self, llm, vector_retriever):
        super().__init__(llm)
        self.vr = vector_retriever

    def retrieve(self, q):
        return self.vr.retrieve(q["question"])


class KGRAG(_Base):
    name = "S2 KG-RAG"

    def __init__(self, llm, graph_retriever):
        super().__init__(llm)
        self.gr = graph_retriever

    def retrieve(self, q):
        return self.gr.retrieve(q["question"])


class Hybrid(_Base):
    name = "S3 Hybrid"

    def __init__(self, llm, graph_retriever, vector_retriever, budget_words: int):
        super().__init__(llm)
        self.gr, self.vr, self.budget = graph_retriever, vector_retriever, budget_words

    def retrieve(self, q):
        # split the SAME total budget between facts and chunks (rule used here: half and half)
        # (budgets are passed per call: never mutate shared retrievers, or the other systems change too)
        g = self.gr.retrieve(q["question"], budget_words=self.budget // 2)
        v = self.vr.retrieve(q["question"], budget_words=self.budget - self.budget // 2)
        text = (g.text + "\n" + v.text).strip()
        return RetrievalResult(text=text, titles=g.titles | v.titles, items=g.items + v.items,
                               debug={"graph": g.debug, "vector": v.debug})


class Oracle(_Base):
    name = "S4 Oracle"

    def __init__(self, llm, docs_by_title: Dict):
        super().__init__(llm)
        self.docs = docs_by_title

    def retrieve(self, q):
        titles = [t for t in q["supporting_titles"] if t in self.docs]
        text = "\n".join("### " + t + "\n" + "\n".join(self.docs[t].sentences) for t in titles)
        return RetrievalResult(text=text, titles=set(titles))


def is_abstain(answer: str) -> bool:
    return INSUFFICIENT in answer.lower()
