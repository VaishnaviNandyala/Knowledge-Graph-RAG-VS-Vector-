# 2WikiMultiHopQA samples used in this project

This folder holds one of the two 2Wiki question sets every team member uses.
**Do not edit these files by hand.** If something must change, tell M1, who regenerates them.

## The two sets
| Folder | What it is | Use it for |
|---|---|---|
| `data/2wiki/` | Random sample of 500 dev questions | Main 2Wiki results (Tables 2 and 3) |
| `data/2wiki_balanced/` | 125 dev questions of each of the 4 types (500 total) | The by-question-type table (Table 4) |

## Where the data comes from
- Dataset: 2WikiMultiHopQA, dev split (`dev.json`)
- Downloaded from: the dataset's GitHub page (Alab-NII/2wikimultihop)
- Licence: [check the licence in that GitHub repository and write it here]
- The raw `dev.json` (about 57 MB) is NOT in the repository. Download it again to regenerate.

## How they were made (reproduce exactly)
```bash
python scripts/convert_2wiki.py --input data/2wiki_raw/dev.json --n 500 --seed 0 --out data/2wiki
python scripts/convert_2wiki.py --input data/2wiki_raw/dev.json --per-type 125 --seed 0 --out data/2wiki_balanced
```
- Date created: [DD-MM-YYYY]   |   Python: [x.y]
- Created by: [your name]

## Files (in each folder)
| File | What is in it |
|---|---|
| `docs.json` | list of `{"title", "sentences"}`: every paragraph used by the sampled questions |
| `questions.json` | list of `{"id", "question", "answer", "supporting_titles", "type", "hops", "evidences"}` |

`evidences` is an extra field: the gold reasoning facts as (subject, relation, object), in Wikidata style.
Other code ignores it.

## Numbers
| Item | `data/2wiki/` (random) | `data/2wiki_balanced/` |
|---|---|---|
| Questions | 500 | 500 |
| Documents (paragraphs) | 3316 | 3374 |
| Sentences | 11999 | 12254 |
| Avg sentences per document | 3.6 | 3.6 |
| compositional | 209 | 125 |
| comparison | 121 | 125 |
| bridge_comparison | 101 | 125 |
| inference | 69 | 125 |
| Hops: 2 / 3 / 4 | 386 / 10 / 104 | 365 / 3 / 132 |

## Things to know (known limits)
- `hops` is ESTIMATED from the number of evidence triples (clipped to 2 to 4). It is not an official label.
  Almost no questions are estimated at 3 hops, so do not draw conclusions about 3-hop questions.
- In the random set, `inference` has only 69 questions, so its confidence interval is wide.
- Documents are merged by title; a paragraph used by several questions appears once.
- `supporting_titles` is at document level. 2Wiki also gives sentence level, which we do not use.
- Comparison answers can be yes/no or an entity name; scoring treats them as plain text.
- Only the paragraphs shipped with each question exist here: this is not the whole of Wikipedia.

## Checks
```bash
python team/check_contracts.py --docs data/2wiki/docs.json --questions data/2wiki/questions.json
python team/check_contracts.py --docs data/2wiki_balanced/docs.json --questions data/2wiki_balanced/questions.json
```
The first block of each must print OK.

## Change log
| Date | Change | By |
|---|---|---|
| [date] | First version: random set and balanced set, seed 0 | [name] |