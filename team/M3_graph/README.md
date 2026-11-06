# M3: Graph and KG-RAG (simple words)

## Your job
Take M2's facts and **build the map** (knowledge graph). Then write the search that **walks the map** to find the facts
needed for a question. This is the main idea of the whole project.

## What is in this folder
```
M3_graph/
  README.md
  run_m3.py                  <- your script
  outputs/
     kg_retrieval.csv            <- how well the graph search finds the needed documents
```
Your shared code (in `src/kgrag/`): `extraction/normalize.py`, `extraction/entities.py`, `graph/kg.py`, `graph/retriever.py`,
and the `KGRAG` and `Hybrid` classes in `systems.py`.

## How the search works (5 steps)
1. Find the names in the question (NER).
2. Match each name to a node on the map (**entity linking**).
3. Walk outwards 1, 2 or 3 steps (**hops**) and collect the facts you pass.
4. Sort the facts: closer and more similar to the question = better.
5. Fill the word budget with the best facts and give them to the LLM.

## Run it
```bash
python team/M3_graph/run_m3.py                      # uses M2's triples.jsonl (or toy facts if it is missing)
python team/M3_graph/run_m3.py --hops 3
python team/M3_graph/run_m3.py --embedder sbert --ner spacy --docs data/hotpot/docs.json --questions data/hotpot/questions.json
```
It prints graph size, the biggest hubs, nodes that might be duplicates, a retrieval test for every question
(no LLM needed), the hop ablation, and checks contract C3.

## What you give to others
| To | What | When |
|---|---|---|
| M5 | a working `GraphRetriever` and the `KGRAG` and `Hybrid` systems | Week 5 |
| M5 | hop and normalisation ablation numbers | Week 6 |
| M2 | a list of wrong or duplicate nodes you notice | any time |

## What you need from others
| From | What | When |
|---|---|---|
| M2 | `triples.jsonl` | Week 3 (small), Week 4 (big). Until then use the toy facts. |
| M4 | the embedder (Sentence-BERT) and the LLM | Week 2 and Week 3 |

## Step by step
1. **Week 1:** run `run_m3.py` on the toy data. Read `graph/kg.py` and `graph/retriever.py` line by line.
2. **Week 2:** add a Sentence-BERT fallback to entity linking (when the exact name is not found, take the nearest node name).
3. **Week 3:** on real triples, look at the "might be the same entity" list. Decide the merge rule. Too many merges = false paths; too few = broken paths.
4. **Week 4:** build the real graph. Find the **hub nodes** (for example "United States", years): they connect unrelated things and add noise. Decide how to rank or filter them.
5. **Week 5:** run S2 and S3 end to end with a real LLM. Test two ways to split the budget in the hybrid.
6. **Week 6:** run the hop ablation (1, 2, 3) and the normalisation ablation.

## Words you must understand
- **Hop:** one step along an edge. 2 hops = friends of friends.
- **Entity linking:** matching a name in the question to a node.
- **Hub:** a node with very many connections.
- **Normalisation:** deciding which names are the same node.

## You are done when
- For every question that the oracle can answer, your retrieval finds the needed facts (check `SuppRecall` in `kg_retrieval.csv`).
- You can explain every failure (wrong link? missing edge? too much noise?).

## Common problems
| Problem | Fix |
|---|---|
| `linked entities: []` | The name in the question is not a node. Check spelling and the linking rule. |
| Recall is 0.67 on 3-hop questions | You need `--hops 3` |
| Too many facts, answer buried | Lower the budget, raise the hop penalty, filter hubs |
| Two nodes for one thing | Tell M2, or fix the merge rule in `normalize.py` |
