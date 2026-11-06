from dataclasses import dataclass, field
from typing import Dict, List, Set


@dataclass
class RetrievalResult:
    text: str                                              # context string handed to the LLM
    titles: Set[str] = field(default_factory=set)          # source documents of what was retrieved
    items: List[Dict] = field(default_factory=list)        # what was retrieved (chunks or triples), for display
    debug: Dict = field(default_factory=dict)              # extra info (linked entities, scores...)
