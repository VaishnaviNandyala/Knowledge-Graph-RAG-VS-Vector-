"""Step 3 of extraction: Relation extraction (read a sentence, output (subject, relation, object) triples).

Both classes expose:  extract_batch(list_of_sentences) -> list (one entry per sentence) of [(s, r, o), ...]
"""
import re
from typing import List, Tuple

Raw = Tuple[str, str, str]


class ToyRelationExtractor:
    """A handful of hand-written regex patterns. It only understands the sentence shapes used in the
    toy corpus ("X was directed by Y." ...). It exists to show the INTERFACE and to let the demo run
    offline. In a real run you replace it with REBEL, a BERT relation classifier, or an LLM prompt."""

    PATTERNS = [
        ("director",       r"^(.+?) was directed by (.+?)\.?$"),
        ("author",         r"^(.+?) was written by (.+?)\.?$"),
        ("composer",       r"^(.+?) was composed by (.+?)\.?$"),
        ("developer",      r"^(.+?) was developed by (.+?)\.?$"),
        ("acquired_by",    r"^(.+?) was acquired by (.+?)(?: in \d{4})?\.?$"),
        ("place_of_birth", r"^(.+?) was born in (.+?)\.?$"),
        ("located_in",     r"^(.+?) is located in (?:the )?(.+?)\.?$"),
        ("release_year",   r"^(.+?) was released in (\d{4})\.?$"),
    ]

    def extract_batch(self, sentences: List[str]) -> List[List[Raw]]:
        out = []
        for s in sentences:
            triples = []
            for rel, pat in self.PATTERNS:
                m = re.match(pat, s.strip())
                if m:
                    triples.append((m.group(1).strip(), rel, m.group(2).strip()))
            out.append(triples)
        return out


class RebelRelationExtractor:
    """Real relation extraction with REBEL (Huguet Cabot & Navigli, 2021): a BART-style
    sequence-to-sequence transformer that READS a sentence and GENERATES its triples as text:
        <triplet> Inception <subj> Christopher Nolan <obj> director
    Setup: pip install transformers torch      (model: Babelscape/rebel-large, ~1.6 GB download)
    Relation names are Wikidata property labels ("place of birth", "director", ...)."""

    def __init__(self, model_name: str = "Babelscape/rebel-large", device: str = "cpu", batch_size: int = 8):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device, self.batch_size = device, batch_size

    @staticmethod
    def parse(text: str):
        """Turn REBEL's special-token output into triples (this is the parser from the model card)."""
        triples, subject, relation, obj, current = [], "", "", "", "x"
        text = text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").strip()
        for token in text.split():
            if token == "<triplet>":
                current = "t"
                if relation:
                    triples.append((subject.strip(), relation.strip(), obj.strip()))
                    relation = ""
                subject = ""
            elif token == "<subj>":
                current = "s"
                if relation:
                    triples.append((subject.strip(), relation.strip(), obj.strip()))
                obj = ""
            elif token == "<obj>":
                current = "o"
                relation = ""
            else:
                if current == "t":
                    subject += " " + token
                elif current == "s":
                    obj += " " + token
                elif current == "o":
                    relation += " " + token
        if subject and relation and obj:
            triples.append((subject.strip(), relation.strip(), obj.strip()))
        return triples

    def extract_batch(self, sentences: List[str]) -> List[List[Raw]]:
        import torch
        results = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i + self.batch_size]
            enc = self.tok(batch, max_length=256, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                gen = self.model.generate(enc["input_ids"], attention_mask=enc["attention_mask"],
                                          max_length=256, length_penalty=0, num_beams=3, num_return_sequences=1)
            for g in self.tok.batch_decode(gen, skip_special_tokens=False):
                results.append([(s, r.replace(" ", "_"), o) for s, r, o in self.parse(g)])
        return results
