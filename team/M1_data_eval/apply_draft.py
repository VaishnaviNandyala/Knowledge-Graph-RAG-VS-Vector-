"""Fill YOUR annotation sheet with the drafted facts (annotation_draft.csv), without touching your sentences.

    python team/M1_data_eval/apply_draft.py
    (defaults: --sheet team/M1_data_eval/outputs/gold_triples.csv  --draft team/M1_data_eval/annotation_draft.csv
               --out   team/M1_data_eval/outputs/gold_triples_draft_filled.csv)

The draft is matched to your sheet by ROW NUMBER, and every row is double-checked (same page title, and the sentence
starts the same way). A row that does not match is NOT filled and is marked MISMATCH, so a wrong sheet can never be
filled by mistake. Your input file is never changed: the result goes to a new file.

Every filled row gets status DRAFT. YOU must read each row against its sentence, fix it if needed, and change the
status to OK. Rows marked '?' in the notes column are the uncertain ones: decide those first.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import in_root                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["doc_title", "sentence", "subject", "relation", "object", "notes", "status"]


def norm(s: str) -> str:
    return " ".join(s.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default=os.path.join(HERE, "outputs", "gold_triples.csv"))
    ap.add_argument("--draft", default=os.path.join(HERE, "annotation_draft.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "outputs", "gold_triples_draft_filled.csv"))
    a = ap.parse_args()
    sheet = list(csv.DictReader(open(in_root(a.sheet), encoding="utf-8-sig")))
    draft_rows = list(csv.DictReader(open(in_root(a.draft), encoding="utf-8-sig")))
    by_row = {}
    for d in draft_rows:
        by_row.setdefault(int(d["row"]), []).append(d)

    out, stats = [], {"filled": 0, "empty (no fact)": 0, "mismatch": 0, "not in draft": 0}
    for i, r in enumerate(sheet, start=1):
        entries = by_row.get(i)
        if not entries:
            out.append({**{k: r.get(k, "") for k in FIELDS[:5]}, "notes": r.get("notes", ""), "status": "TODO"})
            stats["not in draft"] += 1
            continue
        e0 = entries[0]
        ok = r["doc_title"].strip() == e0["doc_title"].strip() and norm(r["sentence"]).startswith(norm(e0["sentence_start"]))
        if not ok:
            print(f"MISMATCH at row {i}: sheet has '{r['doc_title']}' / '{r['sentence'][:30]}...' but the draft was made for "
                  f"'{e0['doc_title']}' / '{e0['sentence_start']}...'. Row left unfilled.")
            out.append({**{k: r.get(k, "") for k in FIELDS[:5]}, "notes": "MISMATCH with draft", "status": "MISMATCH"})
            stats["mismatch"] += 1
            continue
        for d in entries:
            out.append({"doc_title": r["doc_title"], "sentence": r["sentence"], "subject": d["subject"],
                        "relation": d["relation"], "object": d["object"], "notes": d["notes"], "status": "DRAFT"})
        has_fact = any(d["subject"].strip() and d["object"].strip() for d in entries)
        stats["filled" if has_fact else "empty (no fact)"] += 1
    out_path = in_root(a.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)
    facts = sum(1 for o in out if o["subject"].strip() and o["object"].strip())
    sents = len({(o["doc_title"], o["sentence"]) for o in out})
    print(f"sheet rows: {len(sheet)} | sentences in output: {sents} | sentences with facts: {stats['filled']} | fact rows: {facts} | "
          f"sentences marked 'no fact': {stats['empty (no fact)']} | mismatches: {stats['mismatch']} | not in draft: {stats['not in draft']}")
    print(f"written: {out_path}")
    print("Next: open it in Excel, check EVERY row against its sentence, fix, set status to OK, then save as gold_triples.csv.")


if __name__ == "__main__":
    main()
