"""Run:  python tests/test_toy.py   (or: pytest tests/)"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from kgrag.config import Config                                  # noqa: E402
from kgrag.evaluation.metrics import bootstrap_ci, supporting_recall  # noqa: E402
from kgrag.extraction.entities import match_entity               # noqa: E402
from kgrag.extraction.relations import RebelRelationExtractor    # noqa: E402
from kgrag.factory import build_everything                       # noqa: E402
from kgrag.text_utils import exact_match, f1_score               # noqa: E402


def test_metrics():
    assert exact_match("The London.", "london") == 1.0
    assert abs(f1_score("New York City", "New York") - 0.8) < 1e-6
    assert supporting_recall(["A", "B"], {"A"}) == 0.5
    lo, hi = bootstrap_ci([1.0] * 10)
    assert lo == hi == 1.0


def test_entity_matching():
    assert match_entity("Birth", ["Birth (film)", "Google"]) == "Birth (film)"
    assert match_entity("the United Kingdom", ["United Kingdom"]) == "United Kingdom"
    assert match_entity("Inception", ["Inception", "Inception (soundtrack)"]) == "Inception"
    assert match_entity("Zzz", ["Google"]) is None


def test_rebel_parser():           # pure string parsing: needs no model download
    out = RebelRelationExtractor.parse("<s><triplet> Inception <subj> Christopher Nolan <obj> director</s>")
    assert out == [("Inception", "director", "Christopher Nolan")]
    two = RebelRelationExtractor.parse("<triplet> A <subj> B <obj> r1 <subj> C <obj> r2")
    assert two == [("A", "r1", "B"), ("A", "r2", "C")]


def test_toy_pipeline_end_to_end():
    env = build_everything(Config())
    facts = {t.as_text() for t in env["triples"]}
    assert "Inception | director | Christopher Nolan" in facts      # coref worked ("It was directed by ...")
    assert "Christopher Nolan | place_of_birth | London" in facts
    qs = {s.name[:2]: s for s in env["systems"]}
    q = {"id": "t", "question": "Where was the director of Inception born?", "answer": "London",
         "supporting_titles": ["Inception", "Christopher Nolan"]}
    assert qs["S2"].run(q).answer == "London"                        # graph walk chains both facts
    assert qs["S4"].run(q).answer == "London"                        # oracle sanity check
    assert qs["S0"].run(q).answer == "insufficient information"      # no retrieval -> no evidence


def test_budget_is_not_mutated_by_hybrid():
    env = build_everything(Config())
    before = (env["graph_retriever"].budget, env["vector_retriever"].budget)
    next(s for s in env["systems"] if s.name.startswith("S3")).run(
        {"id": "t", "question": "Where was the director of Inception born?", "answer": "", "supporting_titles": []})
    assert before == (env["graph_retriever"].budget, env["vector_retriever"].budget)


def test_convert_2wiki():          # converter works on a small sample in the expected 2Wiki format
    import importlib.util
    import json
    import tempfile
    spec = importlib.util.spec_from_file_location("convert_2wiki", os.path.join("scripts", "convert_2wiki.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as tmp:
        docs, questions, skipped = mod.convert(os.path.join("tests", "fixtures", "2wiki_sample.json"), tmp, n=3)
        assert skipped == 0 and len(questions) == 3
        assert {q["type"] for q in questions} == {"compositional", "comparison", "bridge_comparison"}
        assert all(set(q["supporting_titles"]) <= set(docs) for q in questions)      # contract C1 holds
        from kgrag.contracts import check_c1
        assert check_c1(os.path.join(tmp, "docs.json"), os.path.join(tmp, "questions.json")) == []


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
