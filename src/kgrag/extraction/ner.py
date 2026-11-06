"""Step 2 of extraction: Named Entity Recognition (find the entities in text).

Both classes expose the same method, so the rest of the code does not care which one is used:
    extract_with_spans(text) -> [(entity_text, start_char, end_char), ...]
    extract(text)            -> [entity_text, ...]
"""
import re
from typing import Iterable, List, Tuple

from .entities import strip_disambiguation


class ToyNER:
    """Dictionary (gazetteer) matcher: finds known names, longest match first.
    This is NOT a real NER model; it only exists so the toy demo runs with zero downloads.
    Use SpacyNER (or a BERT NER you fine-tune on CoNLL-2003) for real experiments."""

    def __init__(self, gazetteer: Iterable[str] = ()):
        self.names = set()
        self.add(gazetteer)

    def add(self, names: Iterable[str]) -> None:
        for n in names:
            if not n or len(n) < 2 or n.isdigit():
                continue
            self.names.add(n)
            stripped = strip_disambiguation(n)
            if stripped:
                self.names.add(stripped)

    def extract_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
        taken, found = [], []
        for name in sorted(self.names, key=len, reverse=True):          # longest match first
            for m in re.finditer(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text):
                if any(m.start() < e and s < m.end() for s, e in taken):  # overlaps a longer match
                    continue
                taken.append((m.start(), m.end()))
                found.append((name, m.start(), m.end()))
        return sorted(found, key=lambda x: x[1])

    def extract(self, text: str) -> List[str]:
        return [e[0] for e in self.extract_with_spans(text)]


class SpacyNER:
    """Real NER using a pretrained spaCy pipeline (a small neural model).
    Setup:  pip install spacy && python -m spacy download en_core_web_sm"""

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy
        self.nlp = spacy.load(model)

    def add(self, names: Iterable[str]) -> None:   # spaCy needs no gazetteer; kept for a common interface
        pass

    def extract_with_spans(self, text: str):
        return [(e.text, e.start_char, e.end_char) for e in self.nlp(text).ents]

    def extract(self, text: str) -> List[str]:
        return [e[0] for e in self.extract_with_spans(text)]
