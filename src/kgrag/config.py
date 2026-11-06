"""One place for every setting. Change a backend name to switch from toy to real models.

backend options
  ner       : "toy" (gazetteer + regex)            | "spacy"
  coref     : "toy" (pronoun -> page subject)      | "fastcoref" | "none" (ablation)
  relation  : "toy" (regex patterns)               | "rebel"
  embedder  : "tfidf" (pure numpy, no download)    | "sbert"
  llm       : "mock" (rule-based stand-in reader)  | "ollama" | "hf"
"""
from dataclasses import dataclass


@dataclass
class Config:
    # data
    docs_path: str = "data/toy/docs.json"
    questions_path: str = "data/toy/questions.json"

    # backends
    ner: str = "toy"
    coref: str = "toy"
    relation: str = "toy"
    embedder: str = "tfidf"
    llm: str = "mock"

    # model names (used only by the real backends)
    spacy_model: str = "en_core_web_sm"
    sbert_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rebel_model: str = "Babelscape/rebel-large"
    ollama_model: str = "mistral"
    hf_model: str = "mistralai/Mistral-7B-Instruct-v0.2"

    # retrieval settings (keep IDENTICAL for every system: this is the controlled comparison)
    token_budget: int = 80        # max words of context given to the LLM
    top_k_chunks: int = 3         # cap on vector chunks (budget also applies)
    chunk_max_words: int = 60
    hops: int = 2                 # k-hop depth for graph retrieval
    hop_penalty: float = 0.15     # prefer closer facts when ranking
    normalize_merge_corporate_suffix: bool = False

    seed: int = 0
