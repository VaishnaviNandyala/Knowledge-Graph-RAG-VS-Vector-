"""Tiny data containers + loaders.

Corpus format (docs.json):      [{"title": "...", "sentences": ["...", "..."]}, ...]
Question format (questions.json): [{"id", "question", "answer", "supporting_titles", "type", "hops"}, ...]

HotpotQA already gives paragraphs as a title plus a list of sentences, so we use
the same shape for our toy data. scripts/convert_hotpotqa.py produces this format.
"""
import json
from dataclasses import dataclass
from typing import List


@dataclass
class Doc:
    title: str
    sentences: List[str]

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


def load_docs(path: str) -> List[Doc]:
    with open(path, encoding="utf-8") as f:
        return [Doc(d["title"], d["sentences"]) for d in json.load(f)]


def load_questions(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
