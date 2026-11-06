# Team Guide (simple words)

## What are we building?
We give a computer a question like **"Where was the director of Inception born?"**
The answer needs TWO facts from two different pages. We test two ways to find them:

- **Vector RAG:** find text that looks similar to the question. (Often misses the second fact.)
- **KG-RAG:** turn the text into a map of facts (a "knowledge graph") and walk along the map. (Can follow the chain.)

We compare them fairly and write down **when each one wins and why**.

## Who does what (one line each)

| Member | Job in one line | Your folder |
|---|---|---|
| **M1** | Get the data, make the scoring correct, make the stats. | `M1_data_eval/` |
| **M2** | Read the text and pull out facts like (Inception, director, Nolan). | `M2_extraction/` |
| **M3** | Build the fact map and write the "walk the map" search. | `M3_graph/` |
| **M4** | Build the "find similar text" search and set up the AI model (LLM). | `M4_vector_llm/` |
| **M5** | Join everything, run the final tests, make the demo and the paper. | `M5_integration/` |

## Who waits for whom
```
M1 (data) ---> M2 (facts) ---> M3 (map search) ---> M5 (final results)
   |                                  ^                    ^
   +----------> M4 (text search + LLM)+--------------------+
```
In words:
1. **M1 gives** the data files to M2 and M4.
2. **M2 gives** the facts file (`triples.jsonl`) to M3.
3. **M4 gives** the LLM and the embedder to M3 and M5.
4. **M3 and M4 give** their search code to M5.
5. **M5 runs everything** and makes the tables.

**Nobody has to wait in Week 1.** Every folder already works with the small toy data. Use the toy data until the real
files arrive, then switch.

## The 4 agreed file formats (the "contracts")
If everyone keeps these formats, the parts always fit together.

| # | What | Made by | Used by |
|---|---|---|---|
| C1 | `docs.json` + `questions.json` | M1 | everyone |
| C2 | `triples.jsonl` (one fact per line) | M2 | M3, M5 |
| C3 | a search with `retrieve(question, budget_words)` | M3 and M4 | M5 |
| C4 | an LLM with `generate(prompt)` | M4 | M3, M5 |

Check them any time (takes 2 seconds): `python team/check_contracts.py`

## How to start (everyone, 10 minutes)
```bash
pip install numpy networkx            # that is all the toy mode needs
cd kgrag_project
python tests/test_toy.py               # should print 5 lines starting with "ok"
python team/check_contracts.py         # should print OK four times
```
Then open YOUR folder, read its `README.md`, and run YOUR script.

## The folders (what is inside)
```
kgrag_project/
  src/kgrag/          <- the shared code (everyone's modules live here)
  data/toy/           <- small sample data (21 documents, 9 questions)
  data/hotpot/        <- real data (M1 will create it)
  scripts/            <- the whole-project demo and experiment scripts
  tests/              <- automatic tests
  team/
    TEAM_GUIDE.md     <- this file
    check_contracts.py
    M1_data_eval/     README.md, run_m1.py, outputs/
    M2_extraction/    README.md, run_m2.py, gold_triples_toy.csv, outputs/
    M3_graph/         README.md, run_m3.py, outputs/
    M4_vector_llm/    README.md, run_m4.py, outputs/
    M5_integration/   README.md, run_m5.py, outputs/
```
Every `outputs/` folder is where YOUR script writes files for the others. Do not put your results in someone else's folder.

## Rules for the whole team
1. **Same LLM, same prompt, same budget** for every system. Otherwise the test is not fair.
2. **Change one thing at a time**, then run again. (If you change 3 things you will not know what helped.)
3. **Do not change shared objects while running.** (We found a real bug of this kind in the sample.)
4. **Save expensive results** (facts file, LLM answers) and share the file. Do not run them again.
5. **Be honest:** if KG-RAG loses, that is still a good result. Explain why.
6. Use **git branches**: `m1-data`, `m2-extraction`, `m3-graph`, `m4-vector`, `m5-integration`. Merge after the tests pass.

## Week plan in one table
| Week | What happens |
|---|---|
| 1 | Everyone runs the toy code. Agree the 4 formats. Install the real tools (spaCy, fastcoref, REBEL, Sentence-BERT, Ollama). |
| 2 | M1 gives real data (50 questions). M4 builds vector search on it. |
| 3 | M2 makes the first real `triples.jsonl`. M4 gives the LLM to everyone. **Meeting 1:** run S0, S1, S4 together. |
| 4 | M2 improves the facts. M3 builds the real graph. M1 finishes the gold annotations. |
| 5 | M3 finishes KG-RAG. **Meeting 2:** run all 5 systems together. |
| 6 | M5 runs the big experiments with 500 questions. Then **freeze the code**. |
| 7 | Everyone labels 20 failures. M5 writes discussion. |
| 8 | Demo, paper, slides, practice talk. |

## Quick table: "what do I run?"
| Member | Command |
|---|---|
| M1 | `python team/M1_data_eval/run_m1.py` |
| M2 | `python team/M2_extraction/run_m2.py` |
| M3 | `python team/M3_graph/run_m3.py` |
| M4 | `python team/M4_vector_llm/run_m4.py` |
| M5 | `python team/M5_integration/run_m5.py` |
| anyone | `python team/check_contracts.py` |

All commands work from any folder, and all work offline with the toy data.
