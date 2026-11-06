"""Step 1 of extraction: Coreference resolution (replace pronouns with the entity they mean).

    "Inception is a film. It was directed by Christopher Nolan."
 -> "Inception is a film. Inception was directed by Christopher Nolan."

Without this, the graph gets nodes called "It" / "He", which connect to nothing useful.
Both classes expose:  resolve(doc) -> list of sentences with pronouns replaced.
"""
import re
from typing import List

from ..data import Doc


class ToyCoref:
    """Wikipedia heuristic: a sentence that STARTS with He/She/It/They talks about the page's topic
    (the document title). Simple and surprisingly effective on encyclopedia text, but it will be
    wrong whenever a pronoun refers to someone else. Real coreference models handle that."""

    PRONOUN = re.compile(r"^(He|She|It|They)\b")

    def resolve(self, doc: Doc) -> List[str]:
        out = []
        for sent in doc.sentences:
            m = self.PRONOUN.match(sent)
            out.append(doc.title + sent[m.end():] if m else sent)
        return out


class NoCoref:
    """Ablation: do nothing. Compare results with ToyCoref/FastCoref to measure what coreference is worth."""

    def resolve(self, doc: Doc) -> List[str]:
        return list(doc.sentences)


class FastCorefResolver:
    """Real coreference with the `fastcoref` library (neural, based on a pretrained transformer).
    Setup: pip install fastcoref spacy
    NOTE: the fastcoref API changes between versions; if this errors, check its README.
    Idea: predict clusters of mentions that refer to the same entity, then replace every pronoun
    mention by the first (most informative) mention of its cluster."""

    PRONOUNS = {"he", "she", "it", "they", "him", "her", "them", "his", "its", "their"}

    def __init__(self, device: str = "cpu"):
        from fastcoref import FCoref
        import spacy
        self.model = FCoref(device=device)
        self.splitter = spacy.blank("en")
        self.splitter.add_pipe("sentencizer")

    def resolve(self, doc: Doc) -> List[str]:
        text = " ".join(doc.sentences)
        pred = self.model.predict(texts=[text])[0]
        clusters = pred.get_clusters(as_strings=False)          # list of lists of (start, end) char spans
        replacements = []
        for cluster in clusters:
            mentions = [(s, e, text[s:e]) for s, e in cluster]
            named = [m for m in mentions if m[2].lower() not in self.PRONOUNS]
            if not named:
                continue
            rep = named[0][2]                                   # first non-pronoun mention
            for s, e, t in mentions:
                if t.lower() in {"he", "she", "it", "they"}:    # only replace plain subject pronouns
                    replacements.append((s, e, rep))
        for s, e, rep in sorted(replacements, reverse=True):    # right-to-left so offsets stay valid
            text = text[:s] + rep + text[e:]
        return [sent.text.strip() for sent in self.splitter(text).sents]
