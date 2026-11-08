# M2: Information Extraction (simple words)

## Your job
Read the text and **pull out facts** like `(Inception, director, Christopher Nolan)`.
These facts become the map (graph) that M3 uses. If your facts are wrong or missing, the whole KG-RAG idea suffers,
so you also **measure how good your facts are**.

## What is in this folder
```
M2_extraction/
  README.md
  run_m2.py                  <- your script
  gold_triples_toy.csv       <- example of the "correct answers" file (format only)
  outputs/
     triples.jsonl               <- YOUR MAIN OUTPUT: one fact per line (M3 reads this)
```
Your shared code (in `src/kgrag/extraction/`): `coref.py`, `ner.py`, `relations.py`, `types.py`, `pipeline.py`.

## The 4 steps (what happens inside)
1. **Coreference:** "It was directed by Nolan" -> "Inception was directed by Nolan".
2. **NER:** find the names (people, places, companies).
3. **Relation extraction:** turn each sentence into `(subject, relation, object)`.
4. **Normalisation:** make "Google" and "Google Inc" one name (this part is shared with M3).

## Run it
```bash
python team/M2_extraction/run_m2.py                      # toy mode, works now
python team/M2_extraction/run_m2.py --docs data/hotpot/docs.json --ner spacy --coref fastcoref --relation rebel --limit-docs 100
python team/M2_extraction/run_m2.py --relation rebel --lenient    # score without caring about relation names
```
It writes `outputs/triples.jsonl`, shows 10 example facts, checks the format (contract C2), and compares with the gold file.

## What you give to others
| To | What | When |
|---|---|---|
| M3 | `outputs/triples.jsonl` (small version first, then full) | Week 3 and Week 4 |
| M5 | extraction scores (precision, recall), time numbers, coref on/off results | Week 6 |

## What you need from others
| From | What | When |
|---|---|---|
| M1 | `data/hotpot/docs.json` | Week 2 |
| M1 | filled `gold_triples_to_fill.csv` (to measure quality) | Week 4 |

## Step by step
1. **Week 1:** set up on Google Colab (free GPU): `pip install spacy fastcoref transformers torch` and `python -m spacy download en_core_web_sm`. Try each tool on 3 paragraphs.
2. **Week 2:** run NER with `--ner spacy` on the 50-question data. Look at mistakes.
3. **Week 3:** run with `--coref fastcoref --relation rebel` on a small set and **give M3 the file**. REBEL is slow, so use a GPU and never run it twice on the same data.
4. **Week 4:** read 30 triples by hand. REBEL's relation names are Wikidata labels ("place of birth"). Decide: keep a short list of useful relations, or rename them. Compute precision and recall with M1's gold file. Start the full run.
5. **Week 5:** compare REBEL with a second way to extract (for example, ask an LLM to list the facts). Note time and quality.
6. **Week 6:** run the coreference ablation: `--coref none` vs real coreference. Report how many facts change.

## Words you must understand
- **Precision:** of the facts I extracted, how many are right?
- **Recall:** of the facts that exist, how many did I find?
- **Triple:** one fact = (subject, relation, object).

## You are done when
- M3 uses your file **without editing it**.
- You can say "X% of my facts are right, I find Y% of all facts, and the typical mistakes are ...".

## Common problems
| Problem | Fix |
|---|---|
| `fastcoref` or REBEL will not install | Test in Week 1. Use another Colab runtime. Backup: the toy extractor or an LLM prompt. |
| Contract check says "subject is a pronoun" | Coreference failed on that sentence; fix or filter it |
| REBEL is too slow | GPU + `--limit-docs` + save the output file + run in the background |
| Precision is very low | Filter by a relation list; look at 30 wrong triples to see why |
