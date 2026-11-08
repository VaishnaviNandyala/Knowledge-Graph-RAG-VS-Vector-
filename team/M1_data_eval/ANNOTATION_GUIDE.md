# How to annotate gold triples

Goal: for each sentence, write the facts the sentence states. M2 compares the extractor's output with your labels.

## Rules
1. **Only what the sentence says.** Do not use your own knowledge.
2. **One fact = one row.** A sentence with two facts gets two rows (copy the doc_title and sentence).
3. **Replace pronouns** with the page title. "He was born in London" on the page "Christopher Nolan"
   -> subject `Christopher Nolan`.
4. **Write names as they appear in the text.** Do not shorten or correct them.
5. **Use short relation names** from the list below. If none fits, write a new short one and put it in `notes`.
6. **No fact? Leave the row empty.** Many sentences have none. That is normal.
7. **Unsure? Write your best guess and put `?` in `notes`.**

## Suggested relation names
`director, author, composer, producer, developer, founder, place_of_birth, place_of_death, date_of_birth,
nationality, occupation, located_in, capital_of, member_of, part_of, instance_of, release_year, genre,
acquired_by, award, spouse, educated_at`

## Examples
| doc_title | sentence | subject | relation | object |
|---|---|---|---|---|
| Inception | It was directed by Christopher Nolan. | Inception | director | Christopher Nolan |
| Inception | It was released in 2010. | Inception | release_year | 2010 |
| Inception | The film follows a thief who steals secrets through dreams. | *(empty: no clear fact)* | | |
| Android Inc. | It was acquired by Google in 2005. | Android Inc. | acquired_by | Google |

## Quality check
Ask one teammate (M2 or M5) to annotate the same 20 sentences without seeing yours.
Count how many of the facts both of you wrote. Write this agreement percentage in the paper's appendix.
If it is low, discuss the disagreements and refine these rules.