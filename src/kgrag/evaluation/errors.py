"""Error analysis helper: writes failed questions to a CSV with an automatic GUESS of the cause
and an empty column for YOUR manual label. The manual label is what goes in the paper (Tables 8 and 9)."""
import csv
import os
from typing import Dict, List


def auto_guess(row: Dict, system_name: str) -> str:
    if row["system"] != system_name or row["em"] == 1.0:
        return ""
    if row["supp_recall"] < 1.0:
        return "retrieval miss: a supporting document is missing from the context"
    if row["abstain"]:
        return "evidence retrieved but model abstained (check noise / reading)"
    return "evidence retrieved but wrong answer (LLM reasoning, or a wrong/noisy fact)"


def write_error_sheet(rows: List[Dict], path: str) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fails = [r for r in rows if r["em"] == 0.0 and not r["system"].startswith("S4") and not r["system"].startswith("S0")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "qid", "question", "gold", "pred", "retrieved_titles", "auto_guess", "MANUAL_LABEL"])
        for r in fails:
            w.writerow([r["system"], r["qid"], r["question"], r["gold"], r["pred"], r["retrieved_titles"],
                        auto_guess(r, r["system"]), ""])
    return len(fails)
