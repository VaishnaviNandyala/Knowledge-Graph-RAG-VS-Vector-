"""Convert 2WikiMultiHopQA into this project's format (same as scripts/convert_hotpotqa.py).

    1. Download the dataset yourself (see the dataset's GitHub page, "Alab-NII/2wikimultihop") and unzip it.
       You need the DEV file, usually called dev.json.
    2. Run:
         python scripts/convert_2wiki.py --input path/to/dev.json --n 500 --seed 0 --out data/2wiki
       or a balanced sample with the same number of questions of each type:
         python scripts/convert_2wiki.py --input path/to/dev.json --per-type 125 --seed 0 --out data/2wiki
    3. Check it:
         python team/check_contracts.py --docs data/2wiki/docs.json --questions data/2wiki/questions.json

Format expected in the raw file (as we understand it: LOOK AT ONE ENTRY of your file and compare):
    {"_id": "...", "type": "compositional" | "comparison" | "inference" | "bridge_comparison",
     "question": "...", "answer": "...",
     "context": [["Title", ["sentence 1", "sentence 2", ...]], ...],
     "supporting_facts": [["Title", sentence_index], ...],
     "evidences": [["subject", "relation", "object"], ...]}
If your file differs, change the field names in `convert_example` below (that is the only place).

Output (the project format):
    docs.json       [{"title", "sentences"}]
    questions.json  [{"id", "question", "answer", "supporting_titles", "type", "hops", "evidences"}]
"hops" is estimated from the number of evidence triples (2 to 4). It is an approximation: check a few by hand.
`evidences` is an extra field (gold facts, Wikidata style). Other code ignores it; M1 can use it to help with annotation.
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict


def convert_example(ex: dict):
    """One raw example -> (list of (title, sentences), question dict)."""
    paragraphs = []
    for title, sents in ex["context"]:
        sents = [s.strip() for s in sents if s.strip()]
        if sents:
            paragraphs.append((title, sents))
    evidences = ex.get("evidences", [])
    q = {
        "id": ex["_id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "supporting_titles": sorted({t for t, _ in ex["supporting_facts"]}),
        "type": ex["type"],
        "hops": min(max(len(evidences), 2), 4),     # approximation, see the note above
        "evidences": evidences,
    }
    return paragraphs, q


def sample(examples, n, per_type, seed):
    rng = random.Random(seed)
    if per_type:
        by_type = defaultdict(list)
        for ex in examples:
            by_type[ex["type"]].append(ex)
        chosen = []
        for t in sorted(by_type):
            chosen += rng.sample(by_type[t], min(per_type, len(by_type[t])))
        return chosen
    return rng.sample(examples, min(n, len(examples)))


def convert(input_path: str, out_dir: str, n: int = 500, per_type: int = 0, seed: int = 0):
    examples = json.load(open(input_path, encoding="utf-8"))
    chosen = sample(examples, n, per_type, seed)
    docs, questions, skipped = {}, [], 0
    for ex in chosen:
        try:
            paragraphs, q = convert_example(ex)
        except KeyError as e:
            skipped += 1
            if skipped == 1:
                print(f"WARNING: an example is missing the field {e}. Compare with the format in this file's docstring.")
            continue
        for title, sents in paragraphs:
            docs.setdefault(title, sents)
        questions.append(q)
    if not questions:
        raise SystemExit("No example could be converted: check the field names in convert_example().")
    os.makedirs(out_dir, exist_ok=True)
    json.dump([{"title": t, "sentences": s} for t, s in docs.items()], open(os.path.join(out_dir, "docs.json"), "w", encoding="utf-8"))
    json.dump(questions, open(os.path.join(out_dir, "questions.json"), "w", encoding="utf-8"), indent=1)
    return docs, questions, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to the raw dev.json")
    ap.add_argument("--n", type=int, default=500, help="random sample size")
    ap.add_argument("--per-type", type=int, default=0, help="instead: this many questions of EACH type")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/2wiki")
    a = ap.parse_args()
    docs, questions, skipped = convert(a.input, a.out, a.n, a.per_type, a.seed)
    print(f"wrote {len(docs)} documents and {len(questions)} questions to {a.out}/  (skipped {skipped})")
    print("by type:", dict(Counter(q["type"] for q in questions)))
    print("by hops:", dict(Counter(q["hops"] for q in questions)))


if __name__ == "__main__":
    main()
