"""Health check for your gold annotation file. Run it as often as you like.

    python team/M1_data_eval/check_gold.py
    python team/M1_data_eval/check_gold.py --gold team/M1_data_eval/outputs/gold_triples.csv --docs data/hotpot/docs.json

It checks: every sentence exists in docs.json (M2's scoring needs this), facts have subject AND object, no pronoun
subjects, how many rows are still DRAFT/TODO (not yet checked by you), duplicates, and relation-name style.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import in_root                                           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PRONOUNS = {"it", "he", "she", "they", "her", "his", "him", "them", "this", "that"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=os.path.join(HERE, "outputs", "gold_triples.csv"))
    ap.add_argument("--docs", default="data/hotpot/docs.json")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(in_root(a.gold), encoding="utf-8-sig")))
    docs = {d["title"]: [s.strip() for s in d["sentences"]] for d in json.load(open(in_root(a.docs), encoding="utf-8"))}
    problems = []
    sentences = {(r["doc_title"], r["sentence"].strip()) for r in rows}
    facts = [r for r in rows if r.get("subject", "").strip() or r.get("object", "").strip()]

    missing = [(t, s[:50]) for t, s in sentences if s not in docs.get(t, [])]
    if missing:
        problems.append(f"{len(missing)} sentences are NOT found in docs.json (M2 cannot score them). First: {missing[0]}. "
                        "Usual cause: Excel changed quotes or special characters; re-save as 'CSV UTF-8'.")
    half = [i + 2 for i, r in enumerate(rows) if bool(r.get("subject", "").strip()) != bool(r.get("object", "").strip())]
    if half:
        problems.append(f"rows with a subject but no object (or the reverse): file lines {half[:10]}")
    pron = [i + 2 for i, r in enumerate(rows) if r.get("subject", "").strip().lower() in PRONOUNS]
    if pron:
        problems.append(f"subject is a pronoun (replace it with the name): file lines {pron[:10]}")
    no_rel = [i + 2 for i, r in enumerate(rows) if r.get("subject", "").strip() and not r.get("relation", "").strip()]
    if no_rel:
        problems.append(f"facts without a relation: file lines {no_rel[:10]}")
    dup = [k for k, c in Counter((r["doc_title"], r["sentence"], r.get("subject", ""), r.get("relation", ""), r.get("object", ""))
                                for r in facts).items() if c > 1]
    if dup:
        problems.append(f"{len(dup)} duplicate fact rows, e.g. {dup[0][2:]}")
    odd_rel = sorted({r["relation"] for r in facts if r.get("relation") and r["relation"] != r["relation"].lower().replace(" ", "_")})
    if odd_rel:
        problems.append(f"relation names not in lower_snake_case: {odd_rel[:8]}")

    status = Counter(r.get("status", "") or "(none)" for r in rows)
    unsure = sum(1 for r in rows if "?" in r.get("notes", ""))
    print(f"rows: {len(rows)} | sentences: {len(sentences)} | fact rows: {len(facts)} | "
          f"sentences with no fact: {len(sentences) - len({(r['doc_title'], r['sentence']) for r in facts})}")
    print("status of rows:", dict(status), "| rows marked '?' (unsure):", unsure)
    print("relations used:", dict(Counter(r["relation"] for r in facts).most_common(8)))
    print()
    if problems:
        print("PROBLEMS:\n- " + "\n- ".join(problems))
    else:
        print("No format problems found.")
    todo = status.get("DRAFT", 0) + status.get("TODO", 0) + status.get("MISMATCH", 0)
    if todo:
        print(f"\nNOT READY: {todo} rows still have status DRAFT/TODO/MISMATCH. Check each one, then set status to OK.")
    else:
        print("\nAll rows are checked (no DRAFT/TODO left). The file is ready to give to M2.")


if __name__ == "__main__":
    main()
