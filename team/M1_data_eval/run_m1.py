"""M1 (Data and Evaluation): check the data, check the metrics, make files for the others.

Run:   python team/M1_data_eval/run_m1.py
Real:  python team/M1_data_eval/run_m1.py --docs data/hotpot/docs.json --questions data/hotpot/questions.json --subset 50
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import header, in_root                                   # noqa: E402

from kgrag.contracts import check_c1                                  # noqa: E402
from kgrag.data import load_docs, load_questions                      # noqa: E402
from kgrag.evaluation.metrics import bootstrap_ci                     # noqa: E402
from kgrag.text_utils import exact_match, f1_score                    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/toy/docs.json")
    ap.add_argument("--questions", default="data/toy/questions.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=5, help="how many questions in the small test subset")
    ap.add_argument("--gold-sentences", type=int, default=20, help="how many sentences to annotate")
    a = ap.parse_args()
    docs_p, q_p = in_root(a.docs), in_root(a.questions)
    os.makedirs(OUT, exist_ok=True)

    header("1. Is my data in the agreed format (contract C1)?")
    problems = check_c1(docs_p, q_p)
    print("OK: format is correct" if not problems else "\n".join(problems))

    header("2. What is in the data? (these numbers go in Table 1 of the paper)")
    docs, qs = load_docs(docs_p), load_questions(q_p)
    n_sent = sum(len(d.sentences) for d in docs)
    print(f"documents: {len(docs)} | sentences: {n_sent} | avg sentences per document: {n_sent / len(docs):.1f}")
    print(f"questions: {len(qs)} | by type: {dict(Counter(q['type'] for q in qs))} | by hops: {dict(Counter(q['hops'] for q in qs))}")

    header("3. Do the metrics behave? (check by hand that these make sense)")
    pairs = [("London", "london"), ("The London.", "London"), ("New York City", "New York"), ("Paris", "London")]
    print(f"{'prediction':16} {'gold':10} {'EM':>4} {'F1':>6}")
    for p, g in pairs:
        print(f"{p:16} {g:10} {exact_match(p, g):>4.0f} {f1_score(p, g):>6.2f}")
    print("-> EM ignores case, punctuation and 'the'. F1 gives partial credit for overlapping words.")

    header("4. Bootstrap confidence interval: why more questions matter")
    rng = random.Random(a.seed)
    for n in (10, 100, 500):
        vals = [1.0 if rng.random() < 0.6 else 0.0 for _ in range(n)]
        lo, hi = bootstrap_ci(vals)
        print(f"{n:>4} questions: accuracy {sum(vals) / n:.2f}, 95% interval ({lo:.2f}, {hi:.2f}), width {hi - lo:.2f}")
    print("-> the interval gets narrower with more questions. With 10 questions you cannot conclude anything.")

    header("5. Files for the others (written to team/M1_data_eval/outputs/)")
    rs = random.Random(a.seed)
    subset = rs.sample(qs, min(a.subset, len(qs)))
    json.dump(subset, open(os.path.join(OUT, "questions_subset.json"), "w"), indent=1)
    open(os.path.join(OUT, "question_ids_used.txt"), "w").write("\n".join(q["id"] for q in qs))
    pool = [(d.title, s) for d in docs for s in d.sentences]
    pick = rs.sample(pool, min(a.gold_sentences, len(pool)))
    with open(os.path.join(OUT, "gold_triples_to_fill.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_title", "sentence", "subject", "relation", "object", "notes"])
        for t, s in pick:
            w.writerow([t, s, "", "", "", ""])
    print(f"- questions_subset.json       ({len(subset)} questions, for quick tests by everyone)")
    print(f"- question_ids_used.txt        ({len(qs)} ids, goes in the paper appendix)")
    print(f"- gold_triples_to_fill.csv     ({len(pick)} sentences: YOU fill subject/relation/object by hand;")
    print("                                 M2 uses it to measure extraction quality)")


if __name__ == "__main__":
    main()
