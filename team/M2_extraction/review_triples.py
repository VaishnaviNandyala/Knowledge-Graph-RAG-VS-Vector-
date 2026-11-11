"""Hand-check about 30 triples (needed for 'X% of my facts are right, typical mistakes are ...').

  python team/M2_extraction/review_triples.py sample            # makes outputs/review_sheet.csv
  (open it in Excel/Sheets: write y or n in 'correct'; for every n write a short 'error_type')
  python team/M2_extraction/review_triples.py score             # precision + error counts
Suggested error_type words: wrong_subject, wrong_object, wrong_relation, pronoun, not_stated, junk_entity
"""
import argparse, csv, json, os, random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "score"])
    ap.add_argument("--triples", default=os.path.join(OUT, "triples.jsonl"))
    ap.add_argument("--sheet", default=os.path.join(OUT, "review_sheet.csv"))
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    if a.mode == "sample":
        rows = [json.loads(l) for l in open(a.triples, encoding="utf-8") if l.strip()]
        random.Random(0).shuffle(rows)
        with open(a.sheet, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["doc_title", "sentence", "subject", "relation", "object", "correct", "error_type"])
            for t in rows[: a.n]:
                w.writerow([t["doc_title"], t["sentence"], t["subject"], t["relation"], t["object"], "", ""])
        print(f"wrote {min(a.n, len(rows))} triples to {a.sheet}. Fill 'correct' (y/n) and 'error_type', then run: score")
    else:
        rows = list(csv.DictReader(open(a.sheet, encoding="utf-8-sig")))
        done = [r for r in rows if r["correct"].strip().lower() in ("y", "n")]
        if not done:
            print("no rows filled in yet"); return
        good = sum(r["correct"].strip().lower() == "y" for r in done)
        print(f"hand-checked {len(done)} triples: {good} correct -> precision {good / len(done):.0%}")
        errs = Counter(r["error_type"].strip() or "(no label)" for r in done if r["correct"].strip().lower() == "n")
        print("typical mistakes:", dict(errs.most_common()))


if __name__ == "__main__":
    main()
