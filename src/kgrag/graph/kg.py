"""The knowledge graph: nodes = entities, edges = relations, every edge remembers its source sentence."""
from typing import Dict, List, Tuple

import networkx as nx

from ..extraction.types import Triple


class KnowledgeGraph:
    def __init__(self, triples: List[Triple]):
        self.G = nx.MultiDiGraph()
        for t in triples:
            self.G.add_edge(t.subject, t.object, relation=t.relation, sentence=t.sentence, doc=t.doc_title)

    @property
    def nodes(self):
        return list(self.G.nodes)

    def stats(self) -> Dict:
        return {"nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges()}

    def k_hop_edges(self, start: str, hops: int) -> List[Tuple[str, str, Dict, int]]:
        """Breadth-first walk. Returns (u, v, edge_data, hop) for every edge within `hops` steps of `start`.
        Edges are followed in BOTH directions, because 'Nolan --director_of--> film' is as useful as the reverse."""
        if start not in self.G:
            return []
        dist, frontier, seen, out = {start: 0}, [start], set(), []
        for d in range(hops):
            nxt = []
            for n in frontier:
                incident = list(self.G.out_edges(n, keys=True, data=True)) + list(self.G.in_edges(n, keys=True, data=True))
                for u, v, k, data in incident:
                    if (u, v, k) in seen:
                        continue
                    seen.add((u, v, k))
                    out.append((u, v, data, d + 1))
                    other = v if u == n else u
                    if other not in dist:
                        dist[other] = d + 1
                        nxt.append(other)
            frontier = nxt
        return out
