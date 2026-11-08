# HotpotQA small set (for testing code only)

50 questions from HotpotQA (distractor setting), seed 0. It is small so that extraction and the LLM run fast.

- Use it to check that your code works. **Never report numbers from this set in the paper.**
- Its questions may overlap with `../hotpot/`.
- All final results use `../hotpot/`, `../2wiki/` and `../2wiki_balanced/`.
- Command: `python scripts/convert_hotpotqa.py --n 50 --seed 0 --out data/hotpot_small`
- Do not edit these files by hand. Ask M1.