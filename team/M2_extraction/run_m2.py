"""M2 (Information Extraction): turn documents into triples and measure how good they are.

Run (toy):  python team/M2_extraction/run_m2.py
Real:       python team/M2_extraction/run_m2.py --docs data/hotpot/docs.json --ner spacy --coref fastcoref --relation rebel
Output:     team/M2_extraction/outputs/triples.jsonl   (M3 reads this file)
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import header, in_root                                   # noqa: E402

from kgrag.config import Config                                       # noqa: E402
from kgrag.contracts import check_c2                                  # noqa: E402
from kgrag.data import load_docs                                      # noqa: E402
from kgrag.extraction.entities import entity_key                      # noqa: E402
from kgrag.extraction.normalize import EntityNormalizer               # noqa: E402
from kgrag.extraction.pipeline import ExtractionPipeline, save_triples  # noqa: E402
from kgrag.factory import make_coref, make_ner, make_relation         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def norm_triple(s, r, o, lenient):
    return (entity_key(s), entity_key(o)) if lenient else (entity_key(s), r.lower().replace(" ", "_"), entity_key(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/toy/docs.json")
    ap.add_argument("--ner", default="toy"); ap.add_argument("--coref", default="toy")
    ap.add_argument("--relation", default="toy")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "triples.jsonl"))
    ap.add_argument("--gold", default=os.path.join(HERE, "gold_triples_toy.csv"),
                    help="CSV with doc_title,subject,relation,object (M1 makes the real one)")
    ap.add_argument("--lenient", action="store_true", help="ignore the relation NAME when scoring (useful for REBEL)")
    ap.add_argument("--limit-docs", type=int, default=None)
    a = ap.parse_args()

    cfg = Config(ner=a.ner, coref=a.coref, relation=a.relation)
    docs = load_docs(in_root(a.docs))[: a.limit_docs]
    titles = [d.title for d in docs]

    header(f"1. Running extraction: ner={a.ner}, coref={a.coref}, relation={a.relation}")
    ner = make_ner(cfg, titles)
    pipe = ExtractionPipeline(make_coref(cfg), ner, make_relation(cfg), EntityNormalizer(preferred=titles))
    t0 = time.time()
    triples, stats = pipe.run(docs)
    secs = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    save_triples(triples, a.out)
    print({k: v for k, v in stats.items() if k != "merges"})
    print(f"time: {secs:.2f}s for {len(docs)} documents ({secs / max(len(docs), 1) * 1000:.0f} ms per document)")
    print(f"saved {len(triples)} triples -> {os.path.relpath(a.out, in_root('.'))}")

    header("2. Look at your triples (always read some by hand!)")
    for t in triples[:10]:
        print(f"  {t.as_text():60}  <- \"{t.sentence}\"")
    print("\nrelations used:", dict(Counter(t.relation for t in triples).most_common(8)))
    print("entity merges done by normalisation:", stats["merges"] or "none")

    header("3. Does my file follow contract C2? (M3 depends on it)")
    problems = check_c2(a.out)
    print("OK: format is correct" if not problems else "\n".join(problems))

    header("4. Quality against the gold file (precision = how many extracted are right; recall = how many gold found)")
    if not os.path.exists(a.gold):
        print(f"(no gold file at {a.gold}: ask M1 for gold_triples_to_fill.csv, fill it, and pass --gold)")
        return
    gold_rows = list(csv.DictReader(open(a.gold, encoding="utf-8")))
    gold_rows = [r for r in gold_rows if r.get("subject") and r.get("object")]
    gold_docs = {r["doc_title"] for r in gold_rows if r.get("doc_title")}
    gold = {norm_triple(r["subject"], r["relation"], r["object"], a.lenient) for r in gold_rows}
    pred = {norm_triple(t.subject, t.relation, t.object, a.lenient) for t in triples if not gold_docs or t.doc_title in gold_docs}
    tp = len(pred & gold)
    p, r = tp / max(len(pred), 1), tp / max(len(gold), 1)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    print(f"mode: {'lenient (relation name ignored)' if a.lenient else 'strict'} | gold triples: {len(gold)} | extracted: {len(pred)}")
    print(f"precision {p:.2f} | recall {r:.2f} | F1 {f1:.2f}      (these numbers go in Table 6 of the paper)")
    missed = sorted(gold - pred)[:6]
    if missed:
        print("examples the extractor MISSED:", missed)
    wrong = sorted(pred - gold)[:6]
    if wrong:
        print("examples the extractor got WRONG or not in gold:", wrong)


if __name__ == "__main__":
    main()
