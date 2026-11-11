"""Coreference ablation: run extraction with --coref none and with real coreference, compare the files.

  python team/M2_extraction/ablation_coref.py --docs data/hotpot_small/docs.json --ner spacy --relation rebel --limit-docs 100 --rename
Anything after the known options is passed to run_m2.py (e.g. --rename, --lenient).
"""
import argparse, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRON = {"it", "he", "she", "they", "this", "that", "him", "her", "them"}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-coref", default="fastcoref", help="the real coreference backend (fastcoref or toy)")
    a, rest = ap.parse_known_args()
    paths = {}
    for name in ("none", a.with_coref):
        out = os.path.join(HERE, "outputs", f"triples_coref_{name}.jsonl")
        paths[name] = out
        cmd = [sys.executable, os.path.join(HERE, "run_m2.py"), "--coref", name, "--out", out,
               "--scores-json", os.path.join(HERE, "outputs", f"scores_coref_{name}.json")] + rest
        print("\n>>>", " ".join(cmd)); subprocess.run(cmd, check=True)
    A, B = load(paths["none"]), load(paths[a.with_coref])
    ka = {(t["subject"], t["relation"], t["object"]) for t in A}
    kb = {(t["subject"], t["relation"], t["object"]) for t in B}
    pa = sum(t["subject"].lower() in PRON for t in A)
    pb = sum(t["subject"].lower() in PRON for t in B)
    print("\n=== COREFERENCE ABLATION (numbers for M5 / the paper) ===")
    print(f"triples: none={len(ka)}  {a.with_coref}={len(kb)}")
    print(f"only without coref: {len(ka - kb)} | only with coref: {len(kb - ka)} | in both: {len(ka & kb)}")
    print(f"triples whose subject is still a pronoun: none={pa}  {a.with_coref}={pb}")
    print("scores per run are in outputs/scores_coref_*.json")


if __name__ == "__main__":
    main()
