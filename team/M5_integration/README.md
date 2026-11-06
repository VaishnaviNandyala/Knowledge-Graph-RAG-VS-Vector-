# M5: Integration, Experiments and Paper (simple words)

## Your job
Make the **four other parts work together**, run the **final experiments**, build the **demo**, and write the **paper and slides**.
You keep the team on schedule.

## What is in this folder
```
M5_integration/
  README.md
  run_m5.py                  <- your script
  outputs/                   <- your script writes here:
     results_tables.md           the tables for the paper (copy into the paper)
     predictions.csv             every answer of every system
     error_analysis_sheet.csv    the failures; each member labels about 20
     llm_cache.json              saved LLM answers
```
Your shared code: `factory.py`, `config.py`, `scripts/run_experiments.py`, `evaluation/errors.py`, `app/streamlit_app.py`, `tests/`.
The paper draft and the slides are also yours.

## Run it
```bash
python team/M5_integration/run_m5.py                 # toy mode: checks contracts, runs all 5 systems, makes tables, runs tests
python team/M5_integration/run_m5.py --docs data/hotpot/docs.json --questions data/hotpot/questions.json \
      --ner spacy --coref fastcoref --relation rebel --embedder sbert --llm ollama --limit 50
python team/check_contracts.py                       # quick format check any time
streamlit run app/streamlit_app.py                   # the demo
```

## What you give to others
| To | What | When |
|---|---|---|
| everyone | the repository, branches, weekly meeting, the plan | Week 1 |
| everyone | `error_analysis_sheet.csv` (each labels ~20 rows) | Week 7 |
| M1 | the predictions, so M1 can make the final statistics | Week 6 |

## What you need from others
| From | What | When |
|---|---|---|
| M1 | data, metrics | Week 2 |
| M2 | `triples.jsonl` | Week 4 |
| M3 | graph retriever, KG-RAG, hybrid | Week 5 |
| M4 | vector retriever, LLM, S0 and S4 | Week 3 |

## Step by step
1. **Week 1:** create the git repository with 5 branches. Everyone runs the toy code and `check_contracts.py`. Set the weekly meeting day.
2. **Weeks 2 to 3:** switch backends one at a time in `config.py`. **Meeting 1 (end of Week 3):** run S0, S1, S4 on the 50 real questions together.
3. **Week 5, Meeting 2:** run all five systems together on one computer. Fix what breaks. Run `run_m5.py` on 50 questions.
4. **Week 6:** run on 500 questions. Run the ablations (ask each owner). Then **freeze the code**: only bug fixes after this.
5. **Week 7:** hand out the error sheet, merge the labels into the failure tables. Write Discussion and Limitations.
6. **Week 8:** Streamlit demo, finish the paper (Abstract last), slides, practice talk.

## The paper
Use the research-paper draft. Each table comes from a run:
| Table | From |
|---|---|
| Datasets and graph size | M1 + M3 output |
| Main results, by hops, by type | `results_tables.md` |
| Ablations | each owner's script (M2 coref, M3 hops/normalisation, M4 embedder) |
| Extraction quality | M2 |
| Cost | M4 |
| Failure tables | the labelled error sheet |

## You are done when
- A fresh copy of the project reproduces **every number in the paper** with the commands in the README.
- All 4 other members can explain the whole architecture in one minute.

## Common problems
| Problem | Fix |
|---|---|
| A contract check fails | The message says which member to ask |
| Results change between runs | Someone changed a shared setting; check the cache and config; LLM must use temperature 0 |
| Someone is late | Use toy data for their part and move on; their buddy takes the smallest piece |
| Too much work for you | Every member writes their own paper section |
