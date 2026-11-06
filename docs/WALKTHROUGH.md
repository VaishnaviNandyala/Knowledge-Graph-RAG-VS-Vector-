# Walkthrough: what is actually happening

This walks through the output of `python scripts/run_demo.py` (saved in `docs/sample_outputs/demo_output.txt`).
Read it next to the code. Everything shown is real output from the toy mode.

**Reminder:** the toy NER/relation extractor/"LLM" are simple stand-ins. The goal here is to understand the
*flow* and the *kinds of failure*, not to get research numbers.

---

## The problem in one example

Question: **"Where was the director of Inception born?"** Two facts are needed, in two different documents:

- Document "Inception": *Inception was directed by Christopher Nolan.*
- Document "Christopher Nolan": *Christopher Nolan was born in London.*

The second fact never mentions Inception, and the question never mentions Nolan. That is a **multi-hop question**.

## The full demo output

```text
==============================================================================
STAGE 1 - INFORMATION EXTRACTION on one document
==============================================================================
Document: Inception

  original : Inception is a science fiction film.
  coref    : Inception is a science fiction film.
  entities : ['Inception']
  triples  : []

  original : It was directed by Christopher Nolan.
  coref    : Inception was directed by Christopher Nolan.
  entities : ['Inception', 'Christopher Nolan']
  triples  : [('Inception', 'director', 'Christopher Nolan')]

  original : It was released in 2010.
  coref    : Inception was released in 2010.
  entities : ['Inception']
  triples  : [('Inception', 'release_year', '2010')]

  original : The film follows a thief who steals secrets through dreams.
  coref    : The film follows a thief who steals secrets through dreams.
  entities : []
  triples  : []

==============================================================================
STAGE 2 - WHOLE CORPUS: triples, normalisation, graph
==============================================================================
{'documents': 21, 'raw_triples': 24, 'final_triples': 24, 'entities_before_normalisation': 30, 'entities_after': 29}
entity merges done by normalisation: [('Android Inc.', ['Android Inc'])]
graph: {'nodes': 29, 'edges': 24}
first 8 triples:
    Inception | director | Christopher Nolan
    Inception | release_year | 2010
    Christopher Nolan | place_of_birth | London
    London | located_in | United Kingdom
    Interstellar | director | Christopher Nolan
    Interstellar | release_year | 2014
    Inception (soundtrack) | composer | Hans Zimmer
    Inception (soundtrack) | release_year | 2010

==============================================================================
STAGE 3 - QUESTION: Where was the director of Inception born?
==============================================================================
[Vector RAG] similarity ranking (top 6 chunks): [('Jonathan Glazer', 0.321), ('Francis Ford Coppola', 0.279), ('Inception (soundtrack)', 0.279), ('Inception', 0.258), ('Jane Austen', 0.117), ('Hans Zimmer', 0.115)]
[Vector RAG] context sent to the LLM:
   ### Jonathan Glazer
   Jonathan Glazer is a British film director.
   He was born in London.
   ### Francis Ford Coppola
   Francis Ford Coppola is an American film director.
   He was born in Detroit.
   ### Inception (soundtrack)
   The Inception soundtrack features music for the film.
   It was composed by Hans Zimmer.
   It was released in 2010.

[KG-RAG] question entities linked to graph nodes: ['Inception'] | unlinked: []
[KG-RAG] 5 candidate facts within 2 hops; context sent to the LLM:
   hop1  Inception --director--> Christopher Nolan
   hop1  Inception --release_year--> 2010
   hop2  Inception (soundtrack) --release_year--> 2010
   hop2  Interstellar --director--> Christopher Nolan
   hop2  Christopher Nolan --place_of_birth--> London

==============================================================================
STAGE 4 - ANSWERS (same LLM, same prompt, same word budget)
==============================================================================
gold answer: London

S0 LLM only      -> insufficient information     supporting docs found: []
S1 Vector RAG    -> insufficient information     supporting docs found: []
S2 KG-RAG        -> London                       supporting docs found: ['Christopher Nolan', 'Inception']
S3 Hybrid        -> London                       supporting docs found: ['Christopher Nolan', 'Inception']
S4 Oracle        -> London                       supporting docs found: ['Christopher Nolan', 'Inception']

NOTE: the LLM here is a rule-based stand-in (MockLLM). The toy numbers illustrate the mechanism; they are NOT research results.
```

## Stage 1: Information extraction (look at the document "Inception")

Each sentence goes through the pipeline:

| Step | What you see in the output | Why it matters |
|---|---|---|
| Coreference | `It was directed by Christopher Nolan.` became `Inception was directed by Christopher Nolan.` | Without it the triple would start with the node "It". Try `python scripts/run_experiments.py --coref none`: KG-RAG drops from 0.78 EM to **0.00** on the toy set, because no node is called "Inception" any more. |
| NER | `entities: ['Inception', 'Christopher Nolan']` | These strings become graph nodes and are how we find the starting node of a question. |
| Relation extraction | `[('Inception', 'director', 'Christopher Nolan')]` | One edge. Sentences with no pattern (like "The film follows a thief...") produce nothing: **facts that are not extracted are lost forever.** |

## Stage 2: The whole corpus -> graph

- `raw_triples: 24`, `final_triples: 24`: duplicates are removed.
- **Normalisation:** the output lists a merge `('Android Inc.', ['Android Inc'])`. The relation extractor cut the final period off the object, so the same company appeared as two strings; normalisation fixed it. If this merge had NOT happened, the path Android -> Android Inc. -> Google would break.
- **A trap:** run `python scripts/run_experiments.py --merge-corporate` (aggressively treat "Inc." as noise). Now "Android Inc." merges into "Android" (the operating system), the graph contains `Android | developer | Android`, and the answer can still come out right *for the wrong reason*. Always inspect the graph, not just the accuracy.

## Stage 3: Retrieval, side by side

**Vector RAG** ranks chunks by word overlap (TF-IDF here). The top chunks were *Jonathan Glazer*, *Francis Ford Coppola* and *Inception (soundtrack)*: they match the words "director" and "born" or "Inception", but they are about the wrong people. The chunk of Christopher Nolan is never retrieved, because it contains none of the question's distinctive words. In this toy even the first hop (the "Inception" chunk) was crowded out by distractors. This is the retrieval problem the project studies: **similarity is not the same as connection.**

> Fairness note: TF-IDF only matches exact words ("born" does not match "birth"), which makes the baseline weak. A real Sentence-BERT baseline understands meaning and will do better. Run `--embedder sbert` and report both. Do not claim a result that only holds for a weak baseline.

**KG-RAG** did something different:
1. NER found `Inception` in the question and linked it to the node "Inception".
2. It walked 2 hops from that node and found 5 candidate facts (hop 1: the director and release year; hop 2: facts about neighbours, including `Christopher Nolan --place_of_birth--> London`).
3. It ranked them (similarity minus a penalty per hop) and filled the word budget.

Notice the noise: `Inception (soundtrack) --release_year--> 2010` came in because both films share the node "2010". Years and other common values act like hubs that connect unrelated things. On big graphs this noise is why ranking and the budget matter.

## Stage 4: Answers

For the original question, the demo's last block showed: S0 and S1 "insufficient information", S2, S3 and S4 "London".

**Hop depth matters.** The question "In which country was the director of Inception born?" needs 3 facts
(director -> birthplace -> country). With `--hops 2` KG-RAG cannot reach "London located_in United Kingdom":

```text
gold answer: United Kingdom

S0 LLM only      -> insufficient information     supporting docs found: []
S1 Vector RAG    -> insufficient information     supporting docs found: []
S2 KG-RAG        -> insufficient information     supporting docs found: ['Christopher Nolan', 'Inception']
S3 Hybrid        -> insufficient information     supporting docs found: ['Christopher Nolan', 'Inception']
S4 Oracle        -> United Kingdom               supporting docs found: ['Christopher Nolan', 'Inception', 'London']
```

With `--hops 3` the graph walk reaches it:

```text
gold answer: United Kingdom

S0 LLM only      -> insufficient information     supporting docs found: []
S1 Vector RAG    -> insufficient information     supporting docs found: []
S2 KG-RAG        -> United Kingdom               supporting docs found: ['Christopher Nolan', 'Inception', 'London']
S3 Hybrid        -> insufficient information     supporting docs found: ['Inception']
S4 Oracle        -> United Kingdom               supporting docs found: ['Christopher Nolan', 'Inception', 'London']
```

This is the hop-depth ablation: `python scripts/run_experiments.py --ablate-hops` gives EM 0.22, 0.78, 1.00 for
1, 2, 3 hops on the toy set. On real data, more hops also add more noise, so the curve need not be monotonic.
That trade-off is a result you report.

Notice that the **hybrid** (S3) failed on the 3-hop question: it gives the graph only half of the budget, so the
needed facts were cut. The budget split is a design choice worth testing.

## The five systems, and why each exists

| System | Gets | Purpose |
|---|---|---|
| S0 | nothing | does the LLM already know the answer? (here the stand-in knows nothing, so 0) |
| S1 Vector RAG | top similar chunks | the standard baseline |
| S2 KG-RAG | facts from a graph walk | what we are testing |
| S3 Hybrid | half facts, half chunks | combination |
| S4 Oracle | the gold paragraphs | if even this fails, the problem is the LLM, not retrieval |

## What the toy numbers say, and what they do not

Toy run (9 questions): S1 EM 0.56, S2 0.78, S3 0.78, S4 1.00. With only 9 questions the 95% confidence intervals are enormous (for example S2: 0.44 to 1.0), so **no conclusion can be drawn**. They only show that each component does what it should. Your real study needs 500+ questions, and you must report intervals.

## Two bugs found while building this (read them, they are the lesson)

1. **Shared state:** the hybrid system first changed `budget` on the shared retrievers, so later questions for the *other* systems silently used a different budget. Fixed by passing the budget as an argument per call. In a controlled comparison, anything shared must be read-only.
2. **Duplicate candidates:** the same fact appeared twice in the hybrid context (once as a graph fact, once in a chunk). Entity matching saw two candidates and refused to choose, so the system abstained even though the evidence was present. Fixed by de-duplicating. Moral: when a system says "insufficient information" but supporting recall is 1.0, suspect a bug in your reading or matching code.

## From toy to real: what changes

| Toy | Real | What you will notice |
|---|---|---|
| gazetteer NER | spaCy / BERT NER | misses and wrong spans |
| title-based pronoun rule | fastcoref | much better, but still wrong sometimes |
| regex patterns | REBEL | many more triples, many wrong; Wikidata relation names |
| string normalisation | + embeddings | decide merging thresholds; wrong merges |
| TF-IDF | Sentence-BERT + FAISS | stronger baseline: the comparison becomes honest |
| MockLLM | Mistral / Llama / API | LLM errors begin to matter: use the oracle S4 to separate them |

## Vocabulary

**Triple**: (subject, relation, object). **Node/edge**: entity/relation in the graph. **Hop**: one edge step. **Coreference**: pronouns and mentions pointing to the same entity. **Normalisation**: deciding which strings are one entity. **Entity linking**: mapping a name in the question to a graph node. **Budget**: the same maximum amount of context for every system. **Oracle**: a system given the right evidence, an upper bound. **Ablation**: remove or change one component to see what it contributes.
