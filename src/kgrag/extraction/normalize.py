"""Step 4 of extraction: Entity normalisation (decide which name strings are ONE graph node).

If "Google" and "Google LLC" become two nodes, any path that passes through one of them breaks.
If two different things get merged into one node, you create false paths. Both errors hurt the
final accuracy, which is exactly what ablation RQ3 in the paper measures.
"""
import difflib
import re
from collections import Counter
from typing import Dict, Iterable, List

from .entities import entity_key

CORPORATE_SUFFIXES = {"inc", "llc", "ltd", "corp", "corporation", "co", "company"}


class EntityNormalizer:
    def __init__(self, preferred: Iterable[str] = (), fuzzy_cutoff: float = 0.93,
                 merge_corporate_suffix: bool = False, use_fuzzy: bool = True):
        self.preferred = set(preferred)          # e.g. document titles: best canonical names
        self.fuzzy_cutoff = fuzzy_cutoff
        self.merge_corporate_suffix = merge_corporate_suffix   # ABLATION: try True and look at what merges!
        self.use_fuzzy = use_fuzzy
        self.merges: List[tuple] = []            # (canonical, [variants]) for inspection

    def _key(self, name: str) -> str:
        k = entity_key(name)
        if self.merge_corporate_suffix:
            k = " ".join(w for w in k.split() if w not in CORPORATE_SUFFIXES)
        return k

    def fit(self, surface_counts: Counter) -> Dict[str, str]:
        """surface form -> canonical form."""
        groups: Dict[str, List[str]] = {}
        for form in surface_counts:
            groups.setdefault(self._key(form), []).append(form)

        # optional fuzzy merge of near-identical keys (typos, small spelling variants)
        if self.use_fuzzy:
            keys = sorted(groups)
            for i, a in enumerate(keys):
                if a not in groups:
                    continue
                for b in keys[i + 1:]:
                    if b in groups and len(a) >= 6 and a[0] == b[0] and \
                            difflib.SequenceMatcher(None, a, b).ratio() >= self.fuzzy_cutoff:
                        groups[a].extend(groups.pop(b))

        mapping, self.merges = {}, []
        for forms in groups.values():
            pref = [f for f in forms if f in self.preferred]
            canon = pref[0] if pref else max(forms, key=lambda f: (surface_counts[f], -len(f)))
            for f in forms:
                mapping[f] = canon
            if len(forms) > 1:
                self.merges.append((canon, sorted(set(forms) - {canon})))
        return mapping
