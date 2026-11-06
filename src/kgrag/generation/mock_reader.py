"""A tiny rule-based stand-in for an LLM, so the toy demo runs offline.

READ THIS: this is NOT a language model. It does what a good LLM does when it is handed the right
evidence: read facts out of the context and chain them. Its only purpose is to let you SEE the effect of
retrieval (did the context contain both facts?) without needing a GPU or an API key. It understands only
the question shapes in data/toy/questions.json. For real experiments use OllamaLLM or HFLLM.
"""
import re
from typing import List, Optional, Tuple

from ..extraction.entities import match_entity
from ..extraction.relations import ToyRelationExtractor
from .prompt import INSUFFICIENT, split_prompt

_PRONOUN = re.compile(r"^(He|She|It|They)\b")
_REL = ToyRelationExtractor()

# question pattern -> relation chain to follow (or "compare")
QUESTION_PATTERNS = [
    (r"^Where was the director of (.+?) born\?$",                      ["director", "place_of_birth"]),
    (r"^In which country was the director of (.+?) born\?$",           ["director", "place_of_birth", "located_in"]),
    (r"^Where was the author of (.+?) born\?$",                        ["author", "place_of_birth"]),
    (r"^Which company acquired the startup that developed (.+?)\?$",   ["developer", "acquired_by"]),
    (r"^Which film was released earlier, (.+?) or (.+?)\?$",           ["compare:release_year"]),
]


def parse_context(context: str) -> List[Tuple[str, str, str]]:
    """Read facts from the context. Handles both 'FACT: s | r | o' lines (KG) and '### Title' + sentences (text)."""
    facts, title = [], None
    for line in context.splitlines():
        line = line.strip()
        if not line or line == "(no context)":
            continue
        if line.startswith("FACT:"):
            parts = [p.strip() for p in line[5:].split("|")]
            if len(parts) >= 3:
                facts.append(tuple(parts[:3]))
        elif line.startswith("### "):
            title = line[4:].strip()
        else:
            m = _PRONOUN.match(line)
            sent = title + line[m.end():] if (m and title) else line
            facts += _REL.extract_batch([sent])[0]
    return facts


def _lookup(facts, entity: str, relation: str) -> Optional[str]:
    subjects = [s for s, r, o in facts if r == relation]
    node = match_entity(entity, subjects)
    if node is None:
        return None
    return next(o for s, r, o in facts if r == relation and s == node)


def answer(prompt: str) -> str:
    context, question = split_prompt(prompt)
    facts = parse_context(context)
    for pattern, chain in QUESTION_PATTERNS:
        m = re.match(pattern, question)
        if not m:
            continue
        if chain[0].startswith("compare:"):
            rel = chain[0].split(":")[1]
            a, b = m.group(1), m.group(2)
            ya, yb = _lookup(facts, a, rel), _lookup(facts, b, rel)
            if ya is None or yb is None:
                return INSUFFICIENT
            return a if int(ya) <= int(yb) else b
        current = m.group(1)
        for rel in chain:                      # follow the chain: each step needs a fact in the context
            current = _lookup(facts, current, rel)
            if current is None:
                return INSUFFICIENT
        return current
    return INSUFFICIENT
