# KG-RAG vs Vector RAG for Multi-Hop QA: sample project

A small, working skeleton of the whole project, so you can **see what is happening** before you scale it up.
Everything runs offline in "toy mode" with only `numpy` and `networkx`. Every toy component has a real
counterpart (spaCy, fastcoref, REBEL, Sentence-BERT, FAISS, an open-source LLM) behind the same interface;
you switch by changing one setting.

> **Be clear about what the toy is.** The toy extractors and the "LLM" are tiny rule-based stand-ins, written so
> the pipeline runs without downloads. The toy numbers show the *mechanism*; they are **not research results**.
> Your real results come from running the real backends on HotpotQA / 2WikiMultiHopQA.

## 1. Run it in 2 minutes

```bash
pip install -r requirements-toy.txt
python scripts/run_demo.py                  # guided walkthrough, one question, every stage printed
python scripts/run_experiments.py --ablate-hops   # all 5 systems x 9 questions + tables + hop ablation
python tests/test_toy.py                    # sanity tests
```
Compare your output with `docs/sample_outputs/`. Then read `docs/WALKTHROUGH.md`.

## 2. What is happening (the whole project in one picture)

```
OFFLINE (done once)
  documents -> [1 coreference] -> [2 NER] -> [3 relation extraction] -> [4 entity normalisation]
                                                                              |
                                                             triples (subject, relation, object)
                                                                    /                    \
                                                         knowledge graph          vector index (chunks + embeddings)

ONLINE (per question)
  S0  nothing                                              -> LLM
  S1  top-k similar chunks                                 -> LLM      (vector RAG, the baseline)
  S2  link entities -> walk k hops in the graph -> facts   -> LLM      (KG-RAG, the proposed method)
  S3  half facts + half chunks                             -> LLM      (hybrid)
  S4  the gold paragraphs                                  -> LLM      (oracle: upper bound for the LLM)
```
The five systems share **the same documents, the same LLM, the same prompt and the same word budget**. Only
the retrieval differs. That is what makes it a controlled experiment.

## 3. Project map (file -> what it does -> syllabus topic)

| File | What it does | Syllabus link | Real replacement |
|---|---|---|---|
| `src/kgrag/extraction/coref.py` | "It was directed by X" -> "Inception was directed by X" | coreference | `FastCorefResolver` (included) |
| `src/kgrag/extraction/ner.py` | finds entities in text and in the question | NER | `SpacyNER` (included) |
| `src/kgrag/extraction/relations.py` | sentence -> (subject, relation, object) | relation extraction, Seq2Seq | `RebelRelationExtractor` (included) |
| `src/kgrag/extraction/normalize.py` | merges name variants into one node | lexical semantics | add embedding-based merging (exercise) |
| `src/kgrag/extraction/pipeline.py` | runs the four steps, keeps provenance | information extraction | - |
| `src/kgrag/graph/kg.py` | the graph + k-hop walk | graphs, BFS | Neo4j (optional) |
| `src/kgrag/graph/retriever.py` | link entities, walk, rank, fill budget | NER, retrieval | - |
| `src/kgrag/retrieval/embedder.py` | text -> vectors | Word2Vec/BERT, distributional semantics | `SbertEmbedder` (included) |
| `src/kgrag/retrieval/chunker.py`, `vector.py` | chunk, index, top-k search | RAG baseline | FAISS used automatically if installed |
| `src/kgrag/generation/prompt.py` | the single shared prompt | prompting | - |
| `src/kgrag/generation/llm.py` | LLM backends | transformers | `OllamaLLM`, `HFLLM` (included) |
| `src/kgrag/generation/mock_reader.py` | rule-based stand-in LLM (toy only) | - | any real LLM |
| `src/kgrag/systems.py` | S0 to S4 | experiment design | - |
| `src/kgrag/evaluation/` | EM, F1, supporting recall, bootstrap CI, error sheet | evaluation | - |
| `scripts/convert_hotpotqa.py` | download + sample HotpotQA | datasets | - |
| `app/streamlit_app.py` | side-by-side demo | - | - |

## 4. From toy to real (what YOU do)

1. **Get a GPU environment** (Google Colab free tier is enough). `pip install -r requirements-full.txt` and
   `python -m spacy download en_core_web_sm`.
2. **Get real data:** `python scripts/convert_hotpotqa.py --n 200 --seed 0 --out data/hotpot`
   (start with 50 to 200 questions; REBEL over thousands of paragraphs is slow on CPU).
3. **Switch backends one at a time** and check the output after each change:
   ```bash
   python scripts/run_experiments.py --docs data/hotpot/docs.json --questions data/hotpot/questions.json \
       --ner spacy --coref fastcoref --relation rebel --embedder sbert --llm ollama --limit 50
   ```
   (`ollama pull mistral` first; or use `--llm hf` on Colab.) Change **one** backend per run so you know what caused what.
4. **Look at the triples** REBEL produces before trusting them (print 30). Its relation names are Wikidata labels
   ("place_of_birth", "director", ...), not the toy names. Expect noise: that is the finding, not a bug.
5. **Scale to 500+ questions**, add the ablations (`--ablate-hops`, `--merge-corporate`, coref on/off), fill the paper tables.
6. **Error analysis:** open `results/error_analysis_sheet.csv`, label ~100 failures by hand in `MANUAL_LABEL`.

## 5. Which output goes in which paper table

| Paper table | Produced by |
|---|---|
| Table 1 dataset/graph statistics | first line of `run_experiments.py` output (graph + extraction stats) |
| Table 2 main results | "Overall" table |
| Table 3 by hops, Table 4 by type | "By number of hops", "By question type" tables |
| Table 5 ablations | `--ablate-hops`, `--merge-corporate`, and rerun with other backends (coref off: `--coref none`) |
| Tables 8 and 9 failure taxonomy | your manual labels in `error_analysis_sheet.csv` |
| Tables 6 and 7 | extraction precision/recall (hand-annotate ~100 sentences) and timing (add `time.time()` around stages) |

## 6. Honest notes (read before you trust anything)

- **Tested here:** toy mode end to end (demo, experiments, tests).
- **NOT run in my environment (no internet / no GPU):** spaCy, fastcoref, REBEL, Sentence-BERT, FAISS, Ollama,
  Hugging Face LLM, Streamlit app, HotpotQA converter. They are written from the libraries' documented usage and
  compile, but expect to fix small version issues (especially `fastcoref` and the `datasets` field names).
- **Two real bugs were found and fixed while building this**, and both silently changed results: (1) the hybrid
  system modified the shared retrievers' budgets, which would have changed the *other* systems' results on later
  questions; (2) duplicate candidates broke entity linking in the hybrid context. Lesson: a controlled comparison
  is only as good as your tests (see `tests/test_toy.py`).
- The toy corpus is small (21 documents, 9 questions) and was written to illustrate retrieval behaviour. With
  n=9 the confidence intervals are huge, and the toy retrievers are far simpler than real ones, so do not quote any
  toy number as evidence about KG-RAG versus vector RAG.
- Supporting-fact recall is computed at **document-title level** (HotpotQA officially uses sentence level).
- The mock LLM only understands the question shapes in `data/toy/questions.json`.

## 7. Exercises (do these to really understand it)

1. Run the demo for every toy question. For each failure, say which stage broke (coref, NER, relation, normalisation, linking, hops, budget).
2. Run `--merge-corporate` and print the Android triples. What happened to the node "Android Inc."? Why can the answer still be right for the wrong reason?
3. Change `--budget` to 40, 80, 300. Who benefits from a bigger budget, vector or graph? Why?
4. Add a new relation pattern (for example "was founded by") to the toy extractor and a toy question that needs it.
5. Run with `--coref none` (the `NoCoref` class) and measure the coreference ablation: how many triples disappear, and which questions break?
6. Implement embedding-based entity merging in `normalize.py` using `SbertEmbedder`, with a similarity threshold, and report a case where it merges wrongly.
7. Replace TF-IDF with SBERT (`--embedder sbert`) on the toy data. Does vector RAG improve? Does it fix the "born vs birth" gap?
