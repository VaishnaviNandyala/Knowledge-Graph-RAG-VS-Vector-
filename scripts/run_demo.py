"""Guided walkthrough: prints what happens at EVERY stage for one question.
Run:  python scripts/run_demo.py            (toy mode, offline)
      python scripts/run_demo.py --question "Where was the director of The Godfather born?"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kgrag.config import Config            # noqa: E402
from kgrag.data import load_questions      # noqa: E402
from kgrag.factory import build_everything  # noqa: E402


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Where was the director of Inception born?")
    ap.add_argument("--doc", default="Inception", help="document to show the extraction steps on")
    ap.add_argument("--hops", type=int, default=2)
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(__file__), "..")
    os.chdir(root)
    cfg = Config(hops=args.hops)
    env = build_everything(cfg)
    gold = next((q for q in load_questions(cfg.questions_path) if q["question"] == args.question), None)
    q = gold or {"id": "custom", "question": args.question, "answer": "?", "supporting_titles": []}

    banner("STAGE 1 - INFORMATION EXTRACTION on one document")
    doc = next(d for d in env["docs"] if d.title == args.doc)
    ex = env["pipeline"].explain_doc(doc)
    print(f"Document: {ex['title']}")
    for o, r, e, t in zip(ex["original"], ex["resolved"], ex["entities"], ex["raw_triples"]):
        print(f"\n  original : {o}\n  coref    : {r}\n  entities : {e}\n  triples  : {t}")

    banner("STAGE 2 - WHOLE CORPUS: triples, normalisation, graph")
    st = env["stats"]
    print({k: v for k, v in st.items() if k != "merges"})
    print("entity merges done by normalisation:", st["merges"] or "none")
    print("graph:", env["kg"].stats())
    print("first 8 triples:")
    for t in env["triples"][:8]:
        print("   ", t.as_text())

    banner(f"STAGE 3 - QUESTION: {q['question']}")
    vr, gr = env["vector_retriever"], env["graph_retriever"]
    v = vr.retrieve(q["question"])
    print("[Vector RAG] similarity ranking (top 6 chunks):", v.debug["ranked_top"])
    print("[Vector RAG] context sent to the LLM:\n" + "\n".join("   " + l for l in v.text.splitlines()))

    g = gr.retrieve(q["question"])
    print("\n[KG-RAG] question entities linked to graph nodes:", g.debug["linked"], "| unlinked:", g.debug["unlinked"])
    print(f"[KG-RAG] {g.debug.get('candidates', 0)} candidate facts within {cfg.hops} hops; context sent to the LLM:")
    for it in g.items:
        print(f"   hop{it['hop']}  {it['subject']} --{it['relation']}--> {it['object']}")

    banner("STAGE 4 - ANSWERS (same LLM, same prompt, same word budget)")
    print(f"gold answer: {q['answer']}\n")
    for s in env["systems"]:
        out = s.run(q)
        print(f"{s.name:16} -> {out.answer:28} supporting docs found: {sorted(out.retrieval.titles & set(q['supporting_titles']))}")
    print("\nNOTE: the LLM here is a rule-based stand-in (MockLLM). The toy numbers illustrate the mechanism; "
          "they are NOT research results.")


if __name__ == "__main__":
    main()
