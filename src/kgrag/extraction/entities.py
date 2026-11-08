"""Helpers for comparing entity names.

Why this file exists: the same real-world thing appears under many strings
("The United Kingdom", "United Kingdom", "UK"). Whether two strings are the same
node in the graph is the single most error-prone decision in a KG pipeline.
"""
import difflib
import re
from typing import Iterable, Optional


def entity_key(name: str) -> str:
    """Cheap canonical form: lowercase, no punctuation, no leading 'the'. Parentheses are kept
    on purpose, so 'Inception' and 'Inception (soundtrack)' stay different entities."""
    s = name.lower().strip()
    s = re.sub(r"[.,;:!?\"']", "", s)
    s = re.sub(r"^the\s+", "", s)
    return " ".join(s.split())


def strip_disambiguation(name: str) -> str:
    """'Birth (film)' -> 'Birth'  (Wikipedia-style disambiguation suffix)."""
    return re.sub(r"\s*\(.*?\)\s*$", "", name).strip()


def match_entity(name: str, candidates: Iterable[str], fuzzy_cutoff: float = 0.88) -> Optional[str]:
    """Entity linking, simplest possible version. Tries, in order:
       1. exact match on entity_key
       2. unique match after removing a '(disambiguation)' suffix from the candidate
       3. fuzzy string match (difflib)
    Returns the matching candidate, or None. A real system would add embedding similarity here."""
    cands = list(dict.fromkeys(candidates))     # de-duplicate, keep order
    key = entity_key(name)
    keys = {c: entity_key(c) for c in cands}

    exact = [c for c in cands if keys[c] == key]
    if exact:
        return exact[0]

    alias = [c for c in cands if entity_key(strip_disambiguation(c)) == key]
    if len(alias) == 1:
        return alias[0]

    best = difflib.get_close_matches(key, list(keys.values()), n=1, cutoff=fuzzy_cutoff)
    if best:
        for c, k in keys.items():
            if k == best[0]:
                return c
    return None
