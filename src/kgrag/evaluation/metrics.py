"""Metrics used in the paper: EM, F1, supporting-fact recall, abstention rate, bootstrap confidence intervals."""
import random
from typing import Dict, List, Sequence, Tuple


def supporting_recall(gold_titles: Sequence[str], retrieved_titles) -> float:
    """Fraction of gold supporting documents that appear in the retrieved context.
    (HotpotQA officially scores at sentence level; title level is a simpler approximation.)"""
    gold = set(gold_titles)
    return len(gold & set(retrieved_titles)) / len(gold) if gold else 1.0


def bootstrap_ci(values: List[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """95% bootstrap confidence interval of the mean: resample the questions with replacement many times."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n_boot))
    return means[int(n_boot * alpha / 2)], means[int(n_boot * (1 - alpha / 2)) - 1]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarise(rows: List[Dict], by: str = None) -> Dict:
    """rows: one dict per (system, question). Returns metrics per system (optionally split by a field)."""
    out = {}
    systems = sorted({r["system"] for r in rows})
    for s in systems:
        srows = [r for r in rows if r["system"] == s]
        groups = {"all": srows}
        if by:
            for g in sorted({str(r[by]) for r in srows}):
                groups[f"{by}={g}"] = [r for r in srows if str(r[by]) == g]
        out[s] = {}
        for gname, grows in groups.items():
            em = [r["em"] for r in grows]
            lo, hi = bootstrap_ci(em)
            out[s][gname] = {"n": len(grows), "EM": mean(em), "EM_CI": (lo, hi),
                             "F1": mean([r["f1"] for r in grows]),
                             "SuppRecall": mean([r["supp_recall"] for r in grows]),
                             "Abstain": mean([float(r["abstain"]) for r in grows])}
    return out
