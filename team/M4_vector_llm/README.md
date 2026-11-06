# M4: Vector RAG and the LLM (simple words)

## Your job
1. Build the **normal search** (find text that looks similar to the question). This is our baseline, and it must be **strong**,
   because a weak baseline would make our new method look better than it really is.
2. Set up the **LLM** (the AI that writes the final answer) so that all systems use the same one.

## What is in this folder
```
M4_vector_llm/
  README.md
  run_m4.py                  <- your script
  outputs/
     llm_cache.json              <- saved LLM answers (so re-running is free)
```
Your shared code (in `src/kgrag/`): `retrieval/embedder.py`, `retrieval/chunker.py`, `retrieval/vector.py`,
`generation/prompt.py`, `generation/llm.py`, `generation/cache.py`, and `NoRetrieval`, `VectorRAG`, `Oracle` in `systems.py`.

## The 3 systems you own
- **S0:** no search at all. The LLM answers alone.
- **S1:** vector RAG (the baseline).
- **S4:** the "oracle": we hand the LLM the correct paragraphs. If even this fails, the LLM is the problem, not the search.

## Run it
```bash
python team/M4_vector_llm/run_m4.py                  # toy mode; run it TWICE and watch the cache
python team/M4_vector_llm/run_m4.py --embedder sbert --llm ollama --docs data/hotpot/docs.json --questions data/hotpot/questions.json --limit 50
python team/M4_vector_llm/run_m4.py --embedder sbert --chunk-words 100 --topk 4     # try other settings
```

## What you give to others
| To | What | When |
|---|---|---|
| M3 | the embedder (Sentence-BERT) | Week 2 |
| M3, M5 | the working LLM + the shared prompt | Week 3 |
| M5 | S0, S1, S4 working on real data | Week 3 |
| M5 | cost numbers (time, words, LLM calls) | Week 6 |

## What you need from others
| From | What | When |
|---|---|---|
| M1 | real `docs.json` and `questions.json` | Week 2 |

## Step by step
1. **Week 1:** `pip install sentence-transformers faiss-cpu`. Install Ollama, then `ollama pull mistral` (or use a Hugging Face model on Colab). Make the LLM answer one question.
2. **Week 2:** run S1 with `--embedder sbert` on the 50 real questions.
3. **Week 3:** **freeze the prompt** (`generation/prompt.py`) and the LLM settings (temperature 0). Tell everyone: nobody changes them after this. The cache already saves answers.
4. **Week 4:** try different chunk sizes and top-k. Pick the best. Write down every setting you tried (the paper must say your baseline was tuned fairly).
5. **Week 5:** try a second embedding model (the "embedder ablation").
6. **Week 6:** collect the cost numbers (index time, ms per question, words per question, LLM calls).

## Words you must understand
- **Embedding:** a list of numbers that represents the meaning of a sentence.
- **Cosine similarity:** how close two embeddings are.
- **Chunk:** a piece of a document that we search over.
- **Top-k:** the k best chunks we keep.
- **Oracle:** the system that gets the right answer paragraphs. It is a ceiling.

## You are done when
- S0, S1 and S4 run on real data for everyone.
- Every system uses your LLM and prompt.
- Re-running costs nothing (check "0 new LLM calls").
- You can say how you tuned the baseline.

## Common problems
| Problem | Fix |
|---|---|
| Ollama not found | Install it, start it, run `ollama pull mistral` |
| LLM too slow | A smaller model, `--limit 50`, and the cache |
| S4 (oracle) is low | Your prompt or answer cleaning is wrong, not the search. Look at 10 answers. |
| Baseline too weak | Use Sentence-BERT, tune chunk size. Never leave it at the default if KG-RAG got tuning. |
