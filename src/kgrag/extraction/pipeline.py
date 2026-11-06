"""Glue: runs the four extraction steps and returns triples with provenance.

    docs --coref--> resolved sentences --relation extraction--> raw triples --normalise--> final triples
"""
import json
from collections import Counter
from typing import Dict, List

from ..data import Doc
from .types import Triple


class ExtractionPipeline:
    def __init__(self, coref, ner, relation, normalizer):
        self.coref, self.ner, self.relation, self.normalizer = coref, ner, relation, normalizer

    # -- used by the demo to show every intermediate stage on one document ---------------------------
    def explain_doc(self, doc: Doc) -> Dict:
        resolved = self.coref.resolve(doc)
        entities = [self.ner.extract(s) for s in resolved]
        raw = self.relation.extract_batch(resolved)
        return {"title": doc.title, "original": doc.sentences, "resolved": resolved,
                "entities": entities, "raw_triples": raw}

    # -- the real run ------------------------------------------------------------------------------
    def run(self, docs: List[Doc]):
        raw: List[Triple] = []
        for doc in docs:
            resolved = self.coref.resolve(doc)
            for sent, triples in zip(resolved, self.relation.extract_batch(resolved)):
                for s, r, o in triples:
                    raw.append(Triple(s, r, o, sent, doc.title))

        counts = Counter()
        for t in raw:
            counts[t.subject] += 1
            counts[t.object] += 1
        mapping = self.normalizer.fit(counts)

        seen, final = set(), []
        for t in raw:
            nt = Triple(mapping[t.subject], t.relation, mapping[t.object], t.sentence, t.doc_title)
            key = (nt.subject, nt.relation, nt.object)
            if key not in seen:                      # drop duplicate facts found in several sentences
                seen.add(key)
                final.append(nt)

        self.ner.add({t.subject for t in final} | {t.object for t in final})   # new names help question NER
        stats = {"documents": len(docs), "raw_triples": len(raw), "final_triples": len(final),
                 "entities_before_normalisation": len(counts), "entities_after": len(set(mapping.values())),
                 "merges": self.normalizer.merges}
        return final, stats


def save_triples(triples: List[Triple], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_triples(path: str) -> List[Triple]:
    with open(path, encoding="utf-8") as f:
        return [Triple(**json.loads(line)) for line in f if line.strip()]
