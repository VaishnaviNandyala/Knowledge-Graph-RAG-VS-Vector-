"""Anyone can run this any time to check the shared file formats (C1 to C4).
    python team/check_contracts.py
    python team/check_contracts.py --triples team/M2_extraction/outputs/triples.jsonl --docs data/hotpot/docs.json --questions data/hotpot/questions.json
"""
import argparse
import os

from _common import header, in_root

from kgrag.config import Config
from kgrag.contracts import check_c1, check_c2, check_c3, check_c4
from kgrag.data import load_docs, load_questions
from kgrag.factory import build_everything

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="data/toy/docs.json"); ap.add_argument("--questions", default="data/toy/questions.json")
    ap.add_argument("--triples", default=os.path.join(HERE, "M2_extraction", "outputs", "triples.jsonl"))
    a = ap.parse_args()
    header("C1 documents and questions (M1)")
    p1 = check_c1(in_root(a.docs), in_root(a.questions)); print("OK" if not p1 else "\n".join(p1))
    header("C2 triples file (M2)")
    p2 = check_c2(a.triples); print("OK" if not p2 else "\n".join(p2))
    cfg = Config(docs_path=a.docs, questions_path=a.questions)
    env = build_everything(cfg, docs=load_docs(in_root(a.docs)))
    q = load_questions(in_root(a.questions))[0]["question"]
    header("C3 retrievers (M3 graph, M4 vector)")
    p3 = check_c3(env["graph_retriever"], q, "GraphRetriever", "M3") + check_c3(env["vector_retriever"], q, "VectorRetriever", "M4")
    print("OK" if not p3 else "\n".join(p3))
    header("C4 LLM (M4)")
    p4 = check_c4(env["llm"]); print("OK" if not p4 else "\n".join(p4))


if __name__ == "__main__":
    main()
