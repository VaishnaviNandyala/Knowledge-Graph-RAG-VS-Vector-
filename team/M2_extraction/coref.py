"""Step 1 of extraction: Coreference resolution (replace pronouns with the entity they mean).

    "Inception is a film. It was directed by Christopher Nolan."
 -> "Inception is a film. Inception was directed by Christopher Nolan."

Without this, the graph gets nodes called "It" / "He", which connect to nothing useful.
Both classes expose: resolve(doc) -> list of sentences with pronouns replaced.
"""

import re
from typing import List

from ..data import Doc


class ToyCoref:
    """Simple heuristic coreference resolution using the document title."""

    PRONOUN = re.compile(r"^(He|She|It|They)\b")

    def resolve(self, doc: Doc) -> List[str]:
        out = []
        for sent in doc.sentences:
            m = self.PRONOUN.match(sent)
            out.append(doc.title + sent[m.end():] if m else sent)
        return out


class NoCoref:
    """Ablation: do nothing."""

    def resolve(self, doc: Doc) -> List[str]:
        return list(doc.sentences)


class FastCorefResolver:
    """Real neural coreference resolution using the FastCoref library."""

    PRONOUNS = {
        "he", "she", "it", "they",
        "him", "her", "them", "his", "its", "their"
    }

    def __init__(self, device: str = "cpu", doc_chunk: int = 64):
        from fastcoref import FCoref
        import fastcoref.coref_models.modeling_fcoref as fcoref_modeling
        import spacy
        from unittest.mock import patch

        # Force eager attention because FastCoref's RoBERTa model
        # does not support the SDPA implementation in this setup.
        original_from_config = fcoref_modeling.AutoModel.from_config

        patched_from_config = staticmethod(
            lambda config, *args, **kwargs: original_from_config(
                config,
                *args,
                attn_implementation="eager",
                **kwargs
            )
        )

        with patch.object(
            fcoref_modeling.AutoModel,
            "from_config",
            patched_from_config
        ):
            self.model = FCoref(device=device)

        self.doc_chunk = doc_chunk
        self.device = device
        self.splitter = spacy.blank("en")
        self.splitter.add_pipe("sentencizer")

    def _rewrite(self, text: str, clusters) -> List[str]:
        replacements = []
        for cluster in clusters:
            mentions = [(s, e, text[s:e]) for s, e in cluster]
            named = [m for m in mentions if m[2].lower() not in self.PRONOUNS]
            if not named:
                continue
            rep = named[0][2]
            for s, e, mention in mentions:
                if mention.lower() in {"he", "she", "it", "they"}:
                    replacements.append((s, e, rep))

        # Replace from right to left to preserve character offsets.
        for s, e, rep in sorted(replacements, reverse=True):
            text = text[:s] + rep + text[e:]
        return [sent.text.strip() for sent in self.splitter(text).sents]

    def resolve(self, doc: Doc) -> List[str]:
        return self.resolve_many([doc])[0]

    def resolve_many(self, docs: List[Doc]) -> List[List[str]]:
        """Batched version: one model call per chunk of documents instead of one per document."""
        out: List[List[str]] = []
        for i in range(0, len(docs), self.doc_chunk):
            part = docs[i:i + self.doc_chunk]
            texts = [" ".join(d.sentences) for d in part]
            preds = self.model.predict(texts=texts, max_tokens_in_batch=20000)
            for text, pred in zip(texts, preds):
                out.append(self._rewrite(text, pred.get_clusters(as_strings=False)))
        return out
