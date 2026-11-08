"""M3 (Graph and KG-RAG): build the graph from M2's triples and test the graph retrieval.

Run (toy):  python team/M3_graph/run_m3.py
Needs:      team/M2_extraction/outputs/triples.jsonl  (M2 makes it; if missing, this script builds toy triples itself)
Real:       python team/M3_graph/run_m3.py --triples <M2's file> --docs data/hotpot/docs.json --questions data/hotpot/questions.json --embedder sbert --ner spacy
"""
import argparse
import csv
import difflib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import header, in_root                                   # noqa: E402

from kgrag.config import Config                                       # noqa: E402
from kgrag.contracts import check_c3                                  # noqa: E402
from kgrag.data import load_docs, load_questions                      # noqa: E402
from kgrag.evaluation.metrics import supporting_recall                # noqa: E402
from kgrag.extraction.entities import entity_key                      # noqa: E402
from kgrag.extraction.pipeline import load_triples                    # noqa: E402
from kgrag.factory import make_embedder, make_ner                     # noqa: E402
from kgrag.graph.kg import KnowledgeGraph                             # noqa: E402
from kgrag.graph.retriever import GraphRetriever                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
M2_FILE = os.path.join(HERE, "..", "M2_extraction", "outputs", "triples.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default=M2_FILE)
    ap.add_argument("--docs", default="data/toy/docs.json")
    ap.add_argument("--questions", default="data/toy/questions.json")
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--embedder", default="tfidf"); ap.add_argument("--ner", default="toy")
    a = ap.parse_args()
    cfg = Config(ner=a.ner, embedder=a.embedder, hops=a.hops, token_budget=a.budget)
    docs, qs = load_docs(in_root(a.docs)), load_questions(in_root(a.questions))

    header("1. Loading triples")
    if os.path.exists(a.triples):
        triples = load_triples(a.triples)
        print(f"loaded {len(triples)} triples from {os.path.relpath(a.triples, in_root('.'))}")
    else:
        print("M2's file not found -> building toy triples myself (ask M2 for the real file later)")
        from kgrag.factory import build_everything
        triples = build_everything(Config())["triples"]

    header("2. The graph (numbers for Table 1 of the paper)")
    kg = KnowledgeGraph(triples)
    G = kg.G
    print(kg.stats(), f"| average connections per node: {2 * G.number_of_edges() / max(G.number_of_nodes(), 1):.2f}")
    hubs = sorted(G.degree, key=lambda x: -x[1])[:5]
    print("biggest hubs (many connections = possible noise):", hubs)
    keys, dupes = {}, []
    for n in G.nodes:
        keys.setdefault(entity_key(n), []).append(n)
    dupes += [v for v in keys.values() if len(v) > 1]
    names = list(G.nodes)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            if x[0] == y[0] and difflib.SequenceMatcher(None, x.lower(), y.lower()).ratio() > 0.85 and [x, y] not in dupes:
                dupes.append([x, y])
    print("nodes that might be the SAME entity (check by hand, tell M2/M3):", dupes or "none found")

    header(f"3. Retrieval test: {len(qs)} questions, {a.hops} hops (no LLM needed here)")
    ner = make_ner(cfg, [d.title for d in docs])
    ner.add(kg.nodes)
    emb = make_embedder(cfg).fit([f"{t.subject} {t.relation} {t.object}" for t in triples])
    gr = GraphRetriever(kg, ner, emb, a.hops, cfg.hop_penalty, a.budget)
    rows = []
    print(f"{'qid':4} {'hops':>4} {'recall':>6}  linked entities -> found supporting docs")
    for q in qs:
        r = gr.retrieve(q["question"])
        rec = supporting_recall(q["supporting_titles"], r.titles)
        rows.append({"qid": q["id"], "hops": q["hops"], "supp_recall": rec, "linked": r.debug.get("linked"),
                     "n_facts": len(r.items)})
        print(f"{q['id']:4} {q['hops']:>4} {rec:>6.2f}  {r.debug.get('linked')} -> {sorted(r.titles & set(q['supporting_titles']))}")
    print(f"\nmean supporting recall: {sum(r['supp_recall'] for r in rows) / len(rows):.2f}")

    header("4. Hop-depth ablation (retrieval only). Table 5 of the paper.")
    for h in (1, 2, 3):
        g2 = GraphRetriever(kg, ner, emb, h, cfg.hop_penalty, a.budget)
        rec = [supporting_recall(q["supporting_titles"], g2.retrieve(q["question"]).titles) for q in qs]
        print(f"hops={h}: mean supporting recall = {sum(rec) / len(rec):.2f}")
    print("-> more hops reach more facts, but also pull in more noise. Find the best depth for YOUR data.")

    header("5. Does my retriever follow contract C3? (M5 depends on it)")
    p = check_c3(gr, qs[0]["question"], "GraphRetriever", "M3")
    print("OK" if not p else "\n".join(p))

    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    with open(os.path.join(HERE, "outputs", "kg_retrieval.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved team/M3_graph/outputs/kg_retrieval.csv")


if __name__ == "__main__":
    main()
