"""System S2 retrieval: KG-RAG.

    question --NER--> entities --link--> graph nodes --k-hop walk--> candidate facts --rank--> top facts --> context
"""
from typing import List

from ..extraction.entities import match_entity
from ..retrieval.types import RetrievalResult
from .kg import KnowledgeGraph


class GraphRetriever:
    def __init__(self, kg: KnowledgeGraph, ner, embedder, hops: int = 2, hop_penalty: float = 0.15,
                 budget_words: int = 120):
        self.kg, self.ner, self.embedder = kg, ner, embedder
        self.hops, self.hop_penalty, self.budget = hops, hop_penalty, budget_words

    def link_entities(self, question: str):
        """Steps 1+2: find entity mentions in the question and map each to a node in the graph."""
        linked, unlinked = [], []
        for ent in self.ner.extract(question):
            node = match_entity(ent, self.kg.nodes)
            (linked if node else unlinked).append(node or ent)
        return list(dict.fromkeys(linked)), unlinked

    def retrieve(self, question: str, budget_words: int = None) -> RetrievalResult:
        budget = self.budget if budget_words is None else budget_words
        linked, unlinked = self.link_entities(question)

        # Step 3: walk the graph from every linked node, keep the closest hop for each distinct fact
        cands = {}
        for node in linked:
            for u, v, data, hop in self.kg.k_hop_edges(node, self.hops):
                key = (u, data["relation"], v)
                if key not in cands or hop < cands[key][0]:
                    cands[key] = (hop, data)
        if not cands:
            return RetrievalResult(text="", debug={"linked": linked, "unlinked": unlinked, "candidates": 0})

        # Step 4: rank candidate facts = similarity to the question minus a penalty for being far away
        keys = list(cands)
        texts = [f"{u} {r.replace('_', ' ')} {v}" for u, r, v in keys]
        sims = self.embedder.encode(texts) @ self.embedder.encode([question])[0]
        scored = sorted(((float(s) - self.hop_penalty * (cands[k][0] - 1), k) for s, k in zip(sims, keys)),
                        key=lambda x: -x[0])

        # Step 5: fill the word budget with the best facts
        lines, items, titles, used = [], [], set(), 0
        for score, (u, r, v) in scored:
            hop, data = cands[(u, r, v)]
            line = f"FACT: {u} | {r} | {v}"
            w = len(line.split())
            if used + w > budget:
                break
            lines.append(line)
            used += w
            titles.add(data["doc"])
            items.append({"subject": u, "relation": r, "object": v, "hop": hop, "score": round(score, 3),
                          "source": f"{data['doc']}: {data['sentence']}"})
        return RetrievalResult(text="\n".join(lines), titles=titles, items=items,
                               debug={"linked": linked, "unlinked": unlinked, "candidates": len(cands), "words_used": used})
