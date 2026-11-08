"""Remember LLM answers on disk, so re-running an experiment is free (and results stay identical)."""
import hashlib
import json
import os


class CachedLLM:
    def __init__(self, llm, path: str):
        self.llm, self.path = llm, path
        self.hits = self.misses = 0
        self.store = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}

    def generate(self, prompt: str) -> str:
        key = hashlib.sha1(prompt.encode("utf-8")).hexdigest()
        if key in self.store:
            self.hits += 1
            return self.store[key]
        self.misses += 1
        self.store[key] = self.llm.generate(prompt)
        return self.store[key]

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        json.dump(self.store, open(self.path, "w", encoding="utf-8"), indent=0)
