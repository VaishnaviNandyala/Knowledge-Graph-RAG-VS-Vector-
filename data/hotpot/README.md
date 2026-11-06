# HotpotQA sample used in this project

This folder holds the questions and documents every team member uses.
**Do not edit these files by hand.** If something must change, tell M1, who regenerates them.

## Where the data comes from
- Dataset: HotpotQA, "distractor" setting, validation split
- Downloaded with: Hugging Face `datasets`, dataset name `hotpotqa/hotpot_qa`
- Licence: [check the licence on the dataset page and write it here]

## How it was made (reproduce exactly)
```bash
pip install datasets
python scripts/convert_hotpotqa.py --n [500] --seed [0] --out data/hotpot
```
- Date created: [DD-MM-YYYY]
- `datasets` version: [x.y.z]   |   Python: [x.y]
- Created by: [your name]

## Files
| File | What is in it |
|---|---|
| `docs.json` | list of `{"title", "sentences"}`: every paragraph used by the sampled questions |
| `questions.json` | list of `{"id", "question", "answer", "supporting_titles", "type", "hops"}` |

## Numbers (from `python team/M1_data_eval/run_m1.py`)
| Item | Value |
|---|---|
| Questions | 500 |
| Documents (paragraphs) | 4901 |
| Sentences | 20091 |
| Avg sentences per document | 4.1 |
| Question types | bridge: 383, comparison: 117 |
| Hops | all 2 |

## Things to know (known limits)
- `hops` is set to 2 for every question: HotpotQA questions are 2-hop by design.
- Documents are merged by title; if two questions share a paragraph it appears once.
- `supporting_titles` is at document level. Official HotpotQA scores supporting facts at sentence level.
- Comparison questions can have yes/no answers; the scoring handles them as plain text.
- Only the paragraphs shipped with each question exist here: this is not the whole of Wikipedia.

## Checks
`python team/check_contracts.py --docs data/hotpot/docs.json --questions data/hotpot/questions.json` must print OK.

## Change log
| Date | Change | By |
|---|---|---|
| [date] | First version, 500 questions, seed 0 | [name] |