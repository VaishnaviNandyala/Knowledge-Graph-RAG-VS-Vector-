"""ONE prompt template for every system. Same wording, same instructions: only the context differs."""

TEMPLATE = """Answer the question using ONLY the context below.
If the context is not enough, reply "insufficient information".
Give a short answer (a few words).

Context:
{context}

Question: {question}
Answer:"""

INSUFFICIENT = "insufficient information"


def build_prompt(context: str, question: str) -> str:
    return TEMPLATE.format(context=context if context.strip() else "(no context)", question=question)


def split_prompt(prompt: str):
    """Inverse of build_prompt (used by the mock LLM)."""
    ctx = prompt.split("Context:\n", 1)[1].rsplit("\n\nQuestion: ", 1)
    question = ctx[1].rsplit("\nAnswer:", 1)[0].strip()
    return ctx[0].strip(), question
