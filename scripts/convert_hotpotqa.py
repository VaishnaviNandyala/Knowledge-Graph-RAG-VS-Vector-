"""Download HotpotQA (distractor setting) and convert a random sample into this project's format.

    pip install datasets
    python scripts/convert_hotpotqa.py --n 200 --seed 0 --out data/hotpot

Produces data/hotpot/docs.json and data/hotpot/questions.json, then run:
    python scripts/run_experiments.py --docs data/hotpot/docs.json --questions data/hotpot/questions.json ...

NOTE: field names below follow the Hugging Face `hotpotqa/hotpot_qa` dataset card; if your `datasets`
version differs, print(ds[0]) and adjust. Keep the question IDs file: the paper must say which questions you used.
"""
import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/hotpot")
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    idx = random.Random(a.seed).sample(range(len(ds)), a.n)

    docs, questions = {}, []
    for i in idx:
        ex = ds[i]
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            docs.setdefault(title, [s.strip() for s in sents if s.strip()])
        questions.append({
            "id": ex["id"], "question": ex["question"], "answer": ex["answer"],
            "supporting_titles": sorted(set(ex["supporting_facts"]["title"])),
            "type": ex["type"],            # "bridge" or "comparison"
            "hops": 2,                     # HotpotQA questions are (almost all) 2-hop
        })

    os.makedirs(a.out, exist_ok=True)
    json.dump([{"title": t, "sentences": s} for t, s in docs.items()], open(os.path.join(a.out, "docs.json"), "w"))
    json.dump(questions, open(os.path.join(a.out, "questions.json"), "w"), indent=1)
    print(f"wrote {len(docs)} documents and {len(questions)} questions to {a.out}/")


if __name__ == "__main__":
    main()
