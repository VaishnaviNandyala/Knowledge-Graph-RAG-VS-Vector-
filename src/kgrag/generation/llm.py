"""LLM backends. All expose:  generate(prompt) -> short answer string.

Use the SAME backend, prompt and settings for every system. Otherwise you are no longer comparing retrieval.
"""
from . import mock_reader


def _clean(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    return text.strip().strip('"').rstrip(".")


class MockLLM:
    """Rule-based stand-in (see mock_reader.py). Offline, deterministic, free."""

    def generate(self, prompt: str) -> str:
        return mock_reader.answer(prompt)


class OllamaLLM:
    """A local open-source model served by Ollama (https://ollama.com).
    Setup:  install Ollama, then run:  ollama pull mistral     (or llama3, qwen2.5, ...)"""

    def __init__(self, model: str = "mistral", url: str = "http://localhost:11434/api/generate"):
        self.model, self.url = model, url

    def generate(self, prompt: str) -> str:
        import requests
        r = requests.post(self.url, json={"model": self.model, "prompt": prompt, "stream": False,
                                          "options": {"temperature": 0}}, timeout=300)
        r.raise_for_status()
        return _clean(r.json()["response"])


class HFLLM:
    """An open model loaded directly with Hugging Face transformers (needs a GPU for 7B models; use Colab).
    Setup: pip install transformers accelerate torch.  Greedy decoding (do_sample=False) = temperature 0."""

    def __init__(self, model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        from transformers import pipeline
        self.pipe = pipeline("text-generation", model=model, device_map="auto")

    def generate(self, prompt: str) -> str:
        out = self.pipe(prompt, max_new_tokens=32, do_sample=False, return_full_text=False)
        return _clean(out[0]["generated_text"])
