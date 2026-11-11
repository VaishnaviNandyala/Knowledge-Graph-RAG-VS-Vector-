"""Rename REBEL's Wikidata relation labels to the short names M1 uses in gold_triples.csv.

REBEL says "place of birth"; M1's gold says place_of_birth.  Labels NOT in RENAME are dropped when
keep_only=True (this raises precision), and counted so you can see what you dropped and extend the map.

HOW TO IMPROVE: run once with --rename, look at the "dropped labels" list printed by run_m2.py,
and add any useful label here. Labels below are standard Wikidata property names that REBEL uses;
check them against your own run (REBEL's label list is the final authority).
"""
from collections import Counter

# REBEL / Wikidata label (lowercase, spaces)  ->  M1 short name
RENAME = {
    "place of birth": "place_of_birth",
    "date of birth": "date_of_birth",
    "place of death": "place_of_death",
    "date of death": "date_of_death",
    "country of citizenship": "nationality",
    "occupation": "occupation",
    "instance of": "instance_of",
    "located in the administrative territorial entity": "located_in",
    "located in or next to body of water": "located_in",
    "country": "located_in",
    "capital of": "capital_of",
    "member of": "member_of",
    "member of sports team": "member_of_team",
    "part of": "part_of",
    "director": "director",
    "author": "author",
    "composer": "composer",
    "producer": "producer",
    "screenwriter": "screenwriter",
    "performer": "performer",
    "developer": "developer",
    "founded by": "founder",
    "genre": "genre",
    "publication date": "release_year",
    "award received": "award",
    "spouse": "spouse",
    "educated at": "educated_at",
    "residence": "residence",
    "cast member": "cast_member_of",
    "publisher": "publisher",
    "employer": "employer",
    "mother": "mother",
    "subsidiary": "subsidiary_of",
    "participant of": "participant_in",
    "owned by": "acquired_by",
}
ALREADY_CLEAN = set(RENAME.values())


class MappedRelation:
    """Wraps any relation extractor (REBEL or toy) and renames / filters its relations."""

    def __init__(self, inner, keep_only: bool = True):
        self.inner, self.keep_only = inner, keep_only
        self.dropped = Counter()          # labels removed because they were not in RENAME
        self.renamed = Counter()

    def extract_batch(self, sentences):
        out = []
        for triples in self.inner.extract_batch(sentences):
            kept = []
            for s, r, o in triples:
                label = r.lower().replace("_", " ").strip()
                if label in RENAME:
                    kept.append((s, RENAME[label], o)); self.renamed[label] += 1
                elif r in ALREADY_CLEAN:
                    kept.append((s, r, o))
                elif self.keep_only:
                    self.dropped[label] += 1
                else:
                    kept.append((s, label.replace(" ", "_"), o))
            out.append(kept)
        return out
