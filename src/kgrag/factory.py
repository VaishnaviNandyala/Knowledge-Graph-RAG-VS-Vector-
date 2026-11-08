"""Builds every component from a Config. This is where "toy" turns into "real"."""
from typing import Dict

from .config import Config
from .data import load_docs
from .extraction.coref import FastCorefResolver, NoCoref, ToyCoref
from .extraction.ner import SpacyNER, ToyNER
from .extraction.normalize import EntityNormalizer
from .extraction.pipeline import ExtractionPipeline
from .extraction.relations import RebelRelationExtractor, ToyRelationExtractor
from .generation.llm import HFLLM, MockLLM, OllamaLLM
from .graph.kg import KnowledgeGraph
from .graph.retriever import GraphRetriever
from .retrieval.chunker import chunk_docs
from .retrieval.embedder import SbertEmbedder, TfidfEmbedder
from .retrieval.vector import VectorRetriever
from .systems import Hybrid, KGRAG, NoRetrieval, Oracle, VectorRAG


def make_ner(cfg: Config, titles):
    return ToyNER(titles) if cfg.ner == "toy" else SpacyNER(cfg.spacy_model)


def make_coref(cfg: Config):
    return {"toy": ToyCoref, "none": NoCoref, "fastcoref": FastCorefResolver}[cfg.coref]()


def make_relation(cfg: Config):
    return ToyRelationExtractor() if cfg.relation == "toy" else RebelRelationExtractor(cfg.rebel_model)


def make_embedder(cfg: Config):
    return TfidfEmbedder() if cfg.embedder == "tfidf" else SbertEmbedder(cfg.sbert_model)


def make_llm(cfg: Config):
    return {"mock": lambda: MockLLM(), "ollama": lambda: OllamaLLM(cfg.ollama_model),
            "hf": lambda: HFLLM(cfg.hf_model)}[cfg.llm]()


def build_everything(cfg: Config, docs=None) -> Dict:
    docs = docs or load_docs(cfg.docs_path)
    titles = [d.title for d in docs]

    # ---- offline indexing -------------------------------------------------------------------------
    ner = make_ner(cfg, titles)
    pipeline = ExtractionPipeline(make_coref(cfg), ner, make_relation(cfg),
                                  EntityNormalizer(preferred=titles,
                                                   merge_corporate_suffix=cfg.normalize_merge_corporate_suffix))
    triples, stats = pipeline.run(docs)
    kg = KnowledgeGraph(triples)

    chunks = chunk_docs(docs, cfg.chunk_max_words)
    embedder = make_embedder(cfg).fit([c.embed_text for c in chunks])
    vr = VectorRetriever(chunks, embedder, cfg.top_k_chunks, cfg.token_budget)
    gr = GraphRetriever(kg, ner, embedder, cfg.hops, cfg.hop_penalty, cfg.token_budget)

    # ---- the five systems share ONE llm ---------------------------------------------------------------
    llm = make_llm(cfg)
    systems = [NoRetrieval(llm), VectorRAG(llm, vr), KGRAG(llm, gr),
               Hybrid(llm, gr, vr, cfg.token_budget), Oracle(llm, {d.title: d for d in docs})]
    return {"docs": docs, "pipeline": pipeline, "triples": triples, "stats": stats, "kg": kg,
            "vector_retriever": vr, "graph_retriever": gr, "systems": systems, "llm": llm, "ner": ner}
