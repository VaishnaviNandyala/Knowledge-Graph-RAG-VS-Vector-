"""M4 (Vector RAG and LLM): the baseline (S1), no-retrieval (S0), oracle (S4), the LLM and its cache.

Run (toy):  python team/M4_vector_llm/run_m4.py          (run it TWICE: the second time everything comes from the cache)
Real:       python team/M4_vector_llm/run_m4.py --docs data/hotpot/docs.json --questions data/hotpot/questions.json \
                --embedder sbert --llm ollama --limit 50
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _common import header, in_root                                   # noqa: E402

from kgrag.config import Config                                       # noqa: E402
from kgrag.contracts import check_c3, check_c4                        # noqa: E402
from kgrag.data import load_docs, load_questions                      # noqa: E402
from kgrag.evaluation.metrics import summarise                        # noqa: E402
from kgrag.evaluation.runner import run_all                           # noqa: E402
from kgrag.factory import make_embedder, make_llm                     # noqa: E402
from kgrag.generation.cache import CachedLLM                          # noqa: E402
from kgrag.retrieval.chunker import chunk_docs                        # noqa: E402
from kgrag.retrieval.vector import VectorRetriever                    # noqa: E402
from kgrag.systems import NoRetrieval, Oracle, VectorRAG              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/toy/docs.json"); ap.add_argument("--questions", default="data/toy/questions.json")
    ap.add_argument("--embedder", default="tfidf"); ap.add_argument("--llm", default="mock")
    ap.add_argument("--budget", type=int, default=120); ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--chunk-words", type=int, default=60)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cache", default=os.path.join(HERE, "outputs", "llm_cache.json"))
    a = ap.parse_args()
    cfg = Config(embedder=a.embedder, llm=a.llm, token_budget=a.budget, top_k_chunks=a.topk, chunk_max_words=a.chunk_words)
    docs, qs = load_docs(in_root(a.docs)), load_questions(in_root(a.questions))[: a.limit]

    header("1. Building the vector index")
    t0 = time.time()
    chunks = chunk_docs(docs, a.chunk_words)
    emb = make_embedder(cfg).fit([c.embed_text for c in chunks])
    vr = VectorRetriever(chunks, emb, a.topk, a.budget)
    t_index = time.time() - t0
    print(f"{len(docs)} documents -> {len(chunks)} chunks | embedder={a.embedder} | index time {t_index:.2f}s")

    header("2. Is the LLM working (contract C4) and the retriever (contract C3)?")
    llm = CachedLLM(make_llm(cfg), a.cache)
    print("LLM:", "OK" if not check_c4(llm) else check_c4(llm))
    print("Retriever:", "OK" if not check_c3(vr, qs[0]["question"], "VectorRetriever", "M4") else check_c3(vr, qs[0]["question"], "VectorRetriever", "M4"))

    header(f"3. Running S0 (nothing), S1 (vector RAG), S4 (oracle) on {len(qs)} questions")
    systems = [NoRetrieval(llm), VectorRAG(llm, vr), Oracle(llm, {d.title: d for d in docs})]
    rows = run_all(systems, qs)
    s = summarise(rows)
    print(f"{'system':16} {'EM':>6} {'F1':>6} {'SuppRecall':>11} {'Abstain':>8}")
    for name, g in s.items():
        m = g["all"]
        print(f"{name:16} {m['EM']:>6.2f} {m['F1']:>6.2f} {m['SuppRecall']:>11.2f} {m['Abstain']:>8.2f}")
    print("-> S4 (oracle) is the ceiling for the LLM. If S4 is low, the LLM or the prompt is the problem, not retrieval.")

    header("4. Cost numbers (Table 7 of the paper)")
    t0 = time.time()
    ctx_words = []
    for q in qs:
        ctx_words.append(len(vr.retrieve(q["question"]).text.split()))
    lat = (time.time() - t0) / max(len(qs), 1) * 1000
    print(f"index time: {t_index:.2f}s | retrieval latency: {lat:.1f} ms/question | context size: {sum(ctx_words) / len(ctx_words):.0f} words (budget {a.budget})")
    print(f"LLM cache: {llm.hits} answers reused, {llm.misses} new LLM calls  (run again: new calls should be 0)")
    llm.save()
    print(f"cache saved: {os.path.relpath(a.cache, in_root('.'))}")


if __name__ == "__main__":
    main()
