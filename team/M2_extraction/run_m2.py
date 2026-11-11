"""M2 (Information Extraction): turn documents into triples and measure how good they are.

Toy:   python team/M2_extraction/run_m2.py
Real:  python team/M2_extraction/run_m2.py --docs data/hotpot_small/docs.json --ner spacy --coref fastcoref \
         --relation rebel --rename --limit-docs 100 --gold team/M1_data_eval/outputs/gold_triples.csv
Output: team/M2_extraction/outputs/triples.jsonl   (M3 reads this file)

Changes vs the original file: reads M1's gold CSV with a BOM (utf-8-sig), --rename/--keep-all (REBEL -> M1 names),
--scores-json (saves the numbers for the report), and prints which relation labels were dropped.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from _common import header, in_root                                   # noqa: E402

from kgrag.config import Config                                       # noqa: E402
from kgrag.contracts import check_c2                                  # noqa: E402
from kgrag.data import load_docs                                      # noqa: E402
from kgrag.extraction.entities import entity_key                      # noqa: E402
from kgrag.extraction.normalize import EntityNormalizer               # noqa: E402
from kgrag.extraction.pipeline import ExtractionPipeline, save_triples  # noqa: E402
from kgrag.factory import make_coref, make_ner, make_relation         # noqa: E402
from relation_map import MappedRelation                               # noqa: E402


def norm_triple(s, r, o, lenient):
    return (entity_key(s), entity_key(o)) if lenient else (entity_key(s), r.lower().replace(" ", "_"), entity_key(o))


def score(triples, gold_rows, lenient):
    gold_rows = [r for r in gold_rows if r.get("subject") and r.get("object")]
    gold_docs = {r["doc_title"] for r in gold_rows if r.get("doc_title")}
    gold = {norm_triple(r["subject"], r["relation"], r["object"], lenient) for r in gold_rows}
    pred = {norm_triple(t.subject, t.relation, t.object, lenient) for t in triples if not gold_docs or t.doc_title in gold_docs}
    tp = len(pred & gold)
    p, r = tp / max(len(pred), 1), tp / max(len(gold), 1)
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(mode="lenient" if lenient else "strict", gold=len(gold), extracted=len(pred), tp=tp,
                precision=round(p, 4), recall=round(r, 4), f1=round(f1, 4)), gold, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/toy/docs.json")
    ap.add_argument("--ner", default="toy"); ap.add_argument("--coref", default="toy")
    ap.add_argument("--relation", default="toy")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "triples.jsonl"))
    ap.add_argument("--gold", default=os.path.join(HERE, "gold_triples_toy.csv"),
                    help="CSV with doc_title,subject,relation,object (M1's real one: team/M1_data_eval/outputs/gold_triples.csv)")
    ap.add_argument("--lenient", action="store_true", help="only report the lenient score (relation name ignored)")
    ap.add_argument("--rename", action="store_true", help="rename REBEL labels to M1 names and DROP labels not in relation_map.py")
    ap.add_argument("--keep-all", action="store_true", help="with --rename: keep unmapped labels too (as snake_case)")
    ap.add_argument("--scores-json", default=None, help="save the scores/time here (for the report)")
    ap.add_argument("--limit-docs", type=int, default=None)
    a = ap.parse_args()

    cfg = Config(ner=a.ner, coref=a.coref, relation=a.relation)
    docs = load_docs(in_root(a.docs))[: a.limit_docs]
    titles = [d.title for d in docs]

    header(f"1. Running extraction: ner={a.ner}, coref={a.coref}, relation={a.relation}, rename={a.rename}")
    ner = make_ner(cfg, titles)
    rel = make_relation(cfg)
    if a.rename:
        rel = MappedRelation(rel, keep_only=not a.keep_all)
    pipe = ExtractionPipeline(make_coref(cfg), ner, rel, EntityNormalizer(preferred=titles))
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
    print("\nrelations used:", dict(Counter(t.relation for t in triples).most_common(10)))
    if a.rename:
        print("dropped labels (not in relation_map.py; add useful ones):", dict(rel.dropped.most_common(15)))
    print("entity merges done by normalisation:", stats["merges"] or "none")

    header("3. Does my file follow contract C2? (M3 depends on it)")
    problems = check_c2(a.out)
    print("OK: format is correct" if not problems else "\n".join(problems))

    header("4. Quality against the gold file")
    result = {"docs": len(docs), "seconds": round(secs, 2), "triples": len(triples),
              "ner": a.ner, "coref": a.coref, "relation": a.relation, "rename": a.rename}
    if not os.path.exists(a.gold):
        print(f"(no gold file at {a.gold})")
    else:
        gold_rows = list(csv.DictReader(open(a.gold, encoding="utf-8-sig")))      # utf-8-sig: M1's file has a BOM
        print("Gold only labels some sentences, so STRICT precision is a lower bound. Lenient + recall are more reliable.")
        for lenient in ([True] if a.lenient else [False, True]):
            sc, gold, pred = score(triples, gold_rows, lenient)
            result[sc["mode"]] = sc
            print(f"{sc['mode']:8} | gold {sc['gold']} | extracted {sc['extracted']} | "
                  f"precision {sc['precision']:.2f} | recall {sc['recall']:.2f} | F1 {sc['f1']:.2f}")
            print("   MISSED examples:", sorted(gold - pred)[:4])
            print("   WRONG/not-in-gold examples:", sorted(pred - gold)[:4])
    if a.scores_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.scores_json)), exist_ok=True)
        json.dump(result, open(a.scores_json, "w"), indent=2)
        print("scores saved ->", a.scores_json)


if __name__ == "__main__":
    main()
