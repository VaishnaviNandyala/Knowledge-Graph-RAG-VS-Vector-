# 2Wiki balanced sample

500 questions from the 2WikiMultiHopQA dev set, with exactly 125 questions of each type
(compositional, comparison, bridge_comparison, inference). Seed 0.

- How it was made and its known limits: see `../2wiki/README.md`.
- Use it ONLY for the by-question-type table (Table 4).
  Use `../2wiki/` (the random sample) for the main results.
- Command: `python scripts/convert_2wiki.py --input data/2wiki_raw/dev.json --per-type 125 --seed 0 --out data/2wiki_balanced`
- Do not edit these files by hand. Ask M1.