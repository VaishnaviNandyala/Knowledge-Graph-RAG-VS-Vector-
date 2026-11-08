"""The unit of knowledge in this project: a (subject, relation, object) triple + where it came from."""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str
    sentence: str      # the (coreference-resolved) sentence the triple was read from = evidence
    doc_title: str     # the document it came from = provenance / citation

    def as_text(self) -> str:
        return f"{self.subject} | {self.relation} | {self.object}"

    def to_dict(self) -> dict:
        return asdict(self)
