"""Run all five systems on all questions and print the results tables used in the paper.
Run:  python scripts/run_experiments.py                       (toy mode)
      python scripts/run_experiments.py --ablate-hops         (also vary the hop depth)
Real data: python scripts/run_experiments.py --docs data/hotpot/docs.json --questions data/hotpot/questions.json \
             --ner spacy --coref fastcoref --relation rebel --embedder sbert --llm ollama
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from kgrag.config import Config                      # noqa: E402
from kgrag.data import load_questions                # noqa: E402
from kgrag.evaluation.errors import write_error_sheet  # noqa: E402
from kgrag.evaluation.metrics import summarise       # noqa: E402
from kgrag.evaluation.runner import run_all, save_rows  # noqa: E402
from kgrag.factory import build_everything           # noqa: E402


def print_table(summary, group="all"):
    print(f"{'system':16} {'n':>4} {'EM':>6} {'EM 95% CI':>14} {'F1':>6} {'SuppRecall':>11} {'Abstain':>8}")
    for s, groups in summary.items():
        if group in groups:
            m = groups[group]
            print(f"{s:16} {m['n']:>4} {m['EM']:>6.2f} {str(tuple(round(x, 2) for x in m['EM_CI'])):>14} "
                  f"{m['F1']:>6.2f} {m['SuppRecall']:>11.2f} {m['Abstain']:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs"); ap.add_argument("--questions")
    ap.add_argument("--ner", default="toy"); ap.add_argument("--coref", default="toy")
    ap.add_argument("--relation", default="toy"); ap.add_argument("--embedder", default="tfidf")
    ap.add_argument("--llm", default="mock"); ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N questions")
    ap.add_argument("--ablate-hops", action="store_true")
    ap.add_argument("--merge-corporate", action="store_true", help="ablation: aggressive Inc/LLC merging")
    a = ap.parse_args()

    cfg = Config(ner=a.ner, coref=a.coref, relation=a.relation, embedder=a.embedder, llm=a.llm,
                 hops=a.hops, token_budget=a.budget, normalize_merge_corporate_suffix=a.merge_corporate)
    if a.docs: cfg.docs_path = a.docs
    if a.questions: cfg.questions_path = a.questions
    questions = load_questions(cfg.questions_path)[: a.limit]

    env = build_everything(cfg)
    print("graph:", env["kg"].stats(), "| extraction:", {k: v for k, v in env["stats"].items() if k != "merges"})
    rows = run_all(env["systems"], questions)
    save_rows(rows, "results/predictions.csv")
    n = write_error_sheet(rows, "results/error_analysis_sheet.csv")

    print("\n=== Overall (Table 2 in the paper) ===")
    s_all = summarise(rows)
    print_table(s_all)
    print("\n=== By number of hops (Table 3) ===")
    s_hops = summarise(rows, by="hops")
    for g in sorted({g for s in s_hops.values() for g in s if g.startswith("hops")}):
        print(f"\n-- {g}")
        print_table(s_hops, g)
    print("\n=== By question type (Table 4) ===")
    s_type = summarise(rows, by="type")
    for g in sorted({g for s in s_type.values() for g in s if g.startswith("type")}):
        print(f"\n-- {g}")
        print_table(s_type, g)
    print(f"\nWrote results/predictions.csv and results/error_analysis_sheet.csv ({n} failed rows to label by hand).")

    if a.ablate_hops:
        print("\n=== Ablation: hop depth for KG-RAG (Table 5) ===")
        for h in (1, 2, 3):
            cfg.hops = h
            env_h = build_everything(cfg)
            kg_sys = [s for s in env_h["systems"] if s.name.startswith("S2")]
            r = run_all(kg_sys, questions)
            s = summarise(r)["S2 KG-RAG"]["all"]
            print(f"hops={h}:  EM={s['EM']:.2f}  SuppRecall={s['SuppRecall']:.2f}")


if __name__ == "__main__":
    main()
