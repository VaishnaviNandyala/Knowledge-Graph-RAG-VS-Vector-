"""Runs every system on every question and records one row per (system, question)."""
import csv
import os
from typing import Dict, List

from ..systems import is_abstain
from ..text_utils import exact_match, f1_score
from .metrics import supporting_recall


def run_all(systems, questions: List[dict], verbose: bool = False) -> List[Dict]:
    rows = []
    for q in questions:
        for sys_ in systems:
            out = sys_.run(q)
            rows.append({
                "system": sys_.name, "qid": q["id"], "question": q["question"], "gold": q["answer"],
                "pred": out.answer, "em": exact_match(out.answer, q["answer"]), "f1": f1_score(out.answer, q["answer"]),
                "supp_recall": supporting_recall(q["supporting_titles"], out.retrieval.titles),
                "abstain": is_abstain(out.answer), "hops": q.get("hops"), "type": q.get("type"),
                "retrieved_titles": "; ".join(sorted(out.retrieval.titles)),
            })
            if verbose:
                print(f"[{sys_.name:14}] {q['id']}  pred={out.answer!r:28} gold={q['answer']!r}")
    return rows


def save_rows(rows: List[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
