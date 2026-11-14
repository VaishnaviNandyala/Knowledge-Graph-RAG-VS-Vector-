"""System S2 retrieval: KG-RAG.

    question --NER--> entities --link--> graph nodes --k-hop walk--> candidate facts --rank--> top facts --> context
"""
from typing import List

import numpy as np

from ..extraction.entities import match_entity
from ..retrieval.types import RetrievalResult
from .kg import KnowledgeGraph


class GraphRetriever:
    def __init__(self, kg: KnowledgeGraph, ner, embedder, hops: int = 2, hop_penalty: float = 0.15,
                 budget_words: int = 120, link_fallback: bool = False, fallback_threshold: float = 0.80,
                 question_fallback_topk: int = 0, question_fallback_threshold: float = 0.40):
        """link_fallback / question_fallback_* are OFF by default, so every existing result stays identical.
        link_fallback: a name from the question that matches no node exactly is linked to the node whose NAME is
                       closest in meaning (embedding cosine >= fallback_threshold).
        question_fallback_topk: if nothing at all got linked, link the question to the k nodes whose names are
                       closest to the whole question (cosine >= question_fallback_threshold)."""
        self.kg, self.ner, self.embedder, self.hops, self.hop_penalty, self.budget = kg, ner, embedder, hops, hop_penalty, budget_words
        self.link_fallback, self.fallback_threshold = link_fallback, fallback_threshold
        self.question_fallback_topk, self.question_fallback_threshold = question_fallback_topk, question_fallback_threshold
        self._link_cache = {}                       # mention -> node (the graph never changes, so this is safe)

    def _node_vectors(self):
        if getattr(self.kg, "_node_vecs", None) is None:
            self.kg._node_names = list(self.kg.nodes)
            self.kg._node_vecs = self.embedder.encode(self.kg._node_names)
        return self.kg._node_names, self.kg._node_vecs

    def _nearest_nodes(self, text: str, k: int, threshold: float):
        names, vecs = self._node_vectors()
        if not names:
            return []
        sims = vecs @ self.embedder.encode([text])[0]
        order = np.argsort(-sims)[:k]
        return [names[i] for i in order if sims[i] >= threshold]

    def _link_one(self, ent: str):
        if ent not in self._link_cache:
            node = match_entity(ent, self.kg.nodes)
            if node is None and self.link_fallback:
                near = self._nearest_nodes(ent, 1, self.fallback_threshold)
                node = near[0] if near else None
            self._link_cache[ent] = node
        return self._link_cache[ent]

    def link_entities(self, question: str):
        """Steps 1+2: find entity mentions in the question and map each to a node in the graph."""
        linked, unlinked = [], []
        for ent in self.ner.extract(question):
            node = self._link_one(ent)
            (linked if node else unlinked).append(node or ent)
        linked = list(dict.fromkeys(linked))
        if not linked and self.question_fallback_topk > 0:
            linked = self._nearest_nodes(question, self.question_fallback_topk, self.question_fallback_threshold)
        return linked, unlinked

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
