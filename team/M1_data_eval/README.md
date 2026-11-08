# M1: Data and Evaluation (simple words)

## Your job
Make sure everyone uses the **same correct data**, and that the **scores are trustworthy**.
Think of yourself as the exam setter and the examiner.

## What is in this folder
```
M1_data_eval/
  README.md                  <- this file
  run_m1.py                  <- your script
  outputs/                   <- your script writes here:
     questions_subset.json       small set of questions for quick tests (give to everyone)
     question_ids_used.txt       list of question IDs (goes in the paper appendix)
     gold_triples_to_fill.csv    sentences YOU annotate by hand (give to M2)
```
Your shared code (in `src/kgrag/`): `data.py`, `text_utils.py`, `evaluation/metrics.py`, `evaluation/runner.py`.
Your data scripts: `scripts/convert_hotpotqa.py` and `scripts/convert_2wiki.py`.

## Run it
```bash
python team/M1_data_eval/run_m1.py
```
It checks the data format, prints dataset numbers, tests the scoring, shows why more questions help, and writes the 3 files above.

## What you give to others
| To | What | When |
|---|---|---|
| everyone | `data/hotpot/docs.json` and `questions.json` (same format as the toy files) | Week 2 |
| everyone | `question_ids_used.txt`, `questions_subset.json` | Week 2 |
| M2 | `gold_triples_to_fill.csv`, **filled in by hand** (about 100 sentences) | Week 4 |
| M5 | the final tables with confidence intervals | Week 6 |

## What you need from others
Nothing at the start. In Week 6 you need the prediction files from M5 to compute the final statistics.

## Step by step
1. Install: `pip install datasets` (needs internet).
2. Get real data: `python scripts/convert_hotpotqa.py --n 200 --seed 0 --out data/hotpot`.
   Start with 50 questions for quick tests; use 500 later. **Never change the seed** after sharing.
2b. **Week 2 to 3:** also convert 2WikiMultiHopQA (download the raw `dev.json` yourself, then
   `python scripts/convert_2wiki.py --input path/to/dev.json --n 500 --seed 0 --out data/2wiki`;
   add `--per-type 125` for a balanced sample). LOOK at one raw entry first: if the field names differ, fix `convert_example()`.
3. Check it: `python team/M1_data_eval/run_m1.py --docs data/hotpot/docs.json --questions data/hotpot/questions.json --subset 50`.
4. Tell the team the data is ready (they copy the same `data/hotpot/` folder).
5. Fill `gold_triples_to_fill.csv`: for each sentence, write the facts it states.
   Example: sentence "Inception was directed by Christopher Nolan." -> subject `Inception`, relation `director`, object `Christopher Nolan`.
   If a sentence has no fact, leave it empty. If it has two facts, add a second row.
6. Week 6: make the final tables (overall, by hops, by question type) with 95% confidence intervals.

## Words you must understand
- **Exact Match (EM):** the answer is exactly right (ignoring case, punctuation, "the").
- **F1:** partial credit for overlapping words.
- **95% confidence interval:** "the true accuracy is probably between these two numbers". More questions make it narrower.

## You are done when
- Every other member uses your files.
- Your EM and F1 match the official HotpotQA script on 20 questions.
- Every number in the paper has a confidence interval.

## Common problems
| Problem | Fix |
|---|---|
| `datasets` field names differ | `print(ds[0])` and adjust `convert_hotpotqa.py` |
| Check says "supporting titles not in docs.json" | The converter dropped a document; re-run it |
| People get different results | They are not using the same seed or same question file |
