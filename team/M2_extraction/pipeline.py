
"""Glue: runs the four extraction steps and returns triples with provenance.

    docs --coref--> resolved sentences --relation extraction--> raw triples --normalise--> final triples
"""

import json
import time
from collections import Counter
from typing import Dict, List

from ..data import Doc
from .types import Triple


class ExtractionPipeline:
    def __init__(self, coref, ner, relation, normalizer):
        self.coref = coref
        self.ner = ner
        self.relation = relation
        self.normalizer = normalizer

    # Used by the demo to show every intermediate stage on one document
    def explain_doc(self, doc: Doc) -> Dict:
        resolved = self.coref.resolve(doc)
        entities = [self.ner.extract(s) for s in resolved]
        raw = self.relation.extract_batch(resolved)

        return {
            "title": doc.title,
            "original": doc.sentences,
            "resolved": resolved,
            "entities": entities,
            "raw_triples": raw
        }

    # Runs the complete extraction pipeline
    def run(self, docs: List[Doc]):
        raw: List[Triple] = []

        # Remove triples with standalone pronoun subjects
        pronouns = {
            "it", "he", "she", "they", "we", "i",
            "you", "this", "that", "these", "those"
        }

        # Coreference for ALL docs first (batched on GPU when the resolver supports it) ...
        t0 = time.time()
        if hasattr(self.coref, "resolve_many"):
            resolved_docs = self.coref.resolve_many(docs)
        else:
            resolved_docs = [self.coref.resolve(d) for d in docs]
        print(f"[pipeline] coref done for {len(docs)} docs in {time.time() - t0:.0f}s", flush=True)

        # ... then relation extraction over big groups of documents, so the model sees full batches
        CHUNK = 200
        t1 = time.time()
        for start in range(0, len(docs), CHUNK):
            d_part = docs[start:start + CHUNK]
            r_part = resolved_docs[start:start + CHUNK]
            flat = [sent for resolved in r_part for sent in resolved]
            extracted = self.relation.extract_batch(flat)
            pos = 0
            for doc, resolved in zip(d_part, r_part):
                for sent, triples in zip(resolved, extracted[pos:pos + len(resolved)]):
                    for s, r, o in triples:
                        if s.strip().lower() in pronouns:       # skip standalone pronoun subjects
                            continue
                        raw.append(Triple(s, r, o, sent, doc.title))
                pos += len(resolved)
            done = min(start + CHUNK, len(docs))
            el = time.time() - t1
            print(f"[pipeline] relations: {done}/{len(docs)} docs | {el:.0f}s elapsed | "
                  f"~{el / done * (len(docs) - done):.0f}s left", flush=True)

        # Count entities before normalisation
        counts = Counter()

        for t in raw:
            counts[t.subject] += 1
            counts[t.object] += 1

        mapping = self.normalizer.fit(counts)

        # Normalise entities and remove duplicate triples
        seen = set()
        final = []

        for t in raw:
            nt = Triple(
                mapping[t.subject],
                t.relation,
                mapping[t.object],
                t.sentence,
                t.doc_title
            )

            key = (nt.subject, nt.relation, nt.object)

            if key not in seen:
                seen.add(key)
                final.append(nt)

        # Add extracted entities to NER
        self.ner.add(
            {t.subject for t in final} |
            {t.object for t in final}
        )

        stats = {
            "documents": len(docs),
            "raw_triples": len(raw),
            "final_triples": len(final),
            "entities_before_normalisation": len(counts),
            "entities_after": len(set(mapping.values())),
            "merges": self.normalizer.merges
        }

        return final, stats


def save_triples(triples: List[Triple], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t.to_dict()) + "\n")


def load_triples(path: str) -> List[Triple]:
    with open(path, encoding="utf-8") as f:
        return [
            Triple(**json.loads(line))
            for line in f
            if line.strip()
        ]
