"""Automatic checks of the four agreed formats (contracts C1 to C4).
If a check fails, the message says what is wrong and who should fix it."""
import json
import os
from typing import List

PRONOUNS = {"it", "he", "she", "they", "this", "that", "him", "her", "them"}


def check_c1(docs_path: str, questions_path: str) -> List[str]:
    """C1 (made by M1): docs.json and questions.json."""
    problems = []
    if not os.path.exists(docs_path):
        return [f"C1 [M1]: file not found: {docs_path}"]
    if not os.path.exists(questions_path):
        return [f"C1 [M1]: file not found: {questions_path}"]
    docs = json.load(open(docs_path, encoding="utf-8"))
    qs = json.load(open(questions_path, encoding="utf-8"))
    titles = set()
    for i, d in enumerate(docs):
        if not isinstance(d.get("title"), str) or not isinstance(d.get("sentences"), list) or not d["sentences"]:
            problems.append(f"C1 [M1]: document #{i} needs a 'title' (text) and a non-empty 'sentences' (list)")
        else:
            titles.add(d["title"])
    for q in qs:
        for key in ("id", "question", "answer", "supporting_titles", "type", "hops"):
            if key not in q:
                problems.append(f"C1 [M1]: question {q.get('id', '?')} is missing '{key}'")
        missing = [t for t in q.get("supporting_titles", []) if t not in titles]
        if missing:
            problems.append(f"C1 [M1]: question {q.get('id')} lists supporting titles not in docs.json: {missing}")
    return problems


def check_c2(triples_path: str) -> List[str]:
    """C2 (made by M2): triples.jsonl."""
    if not os.path.exists(triples_path):
        return [f"C2 [M2]: file not found: {triples_path}"]
    problems, n = [], 0
    with open(triples_path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                problems.append(f"C2 [M2]: line {ln} is not valid JSON")
                continue
            for key in ("subject", "relation", "object", "sentence", "doc_title"):
                if not str(t.get(key, "")).strip():
                    problems.append(f"C2 [M2]: line {ln} has empty '{key}'")
            if str(t.get("subject", "")).lower() in PRONOUNS:
                problems.append(f"C2 [M2]: line {ln} subject is a pronoun ('{t['subject']}'): coreference did not work here")
    if n == 0:
        problems.append("C2 [M2]: the triples file is empty")
    return problems[:30]


def check_c3(retriever, question: str, name: str, owner: str) -> List[str]:
    """C3 (made by M3 for the graph, M4 for vectors): retrieve(question, budget_words=...)."""
    try:
        r = retriever.retrieve(question, budget_words=60)
    except TypeError:
        return [f"C3 [{owner}]: {name}.retrieve must accept the argument budget_words"]
    problems = []
    for attr in ("text", "titles", "items", "debug"):
        if not hasattr(r, attr):
            problems.append(f"C3 [{owner}]: {name}.retrieve result has no '{attr}'")
    if hasattr(r, "text") and len(r.text.split()) > 60 + 25:
        problems.append(f"C3 [{owner}]: {name} used far more than the word budget ({len(r.text.split())} words > 60)")
    return problems


def check_c4(llm) -> List[str]:
    """C4 (made by M4): generate(prompt) -> str."""
    try:
        out = llm.generate("Answer the question using ONLY the context below.\n\nContext:\n(no context)\n\nQuestion: test?\nAnswer:")
    except Exception as e:                                   # noqa: BLE001
        return [f"C4 [M4]: llm.generate crashed: {e}"]
    return [] if isinstance(out, str) else ["C4 [M4]: llm.generate must return a string"]
