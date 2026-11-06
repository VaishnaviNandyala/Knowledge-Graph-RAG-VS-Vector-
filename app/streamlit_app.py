"""Demo app:  streamlit run app/streamlit_app.py
Side-by-side answers from Vector RAG, KG-RAG and Hybrid, with the evidence each one used."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st                      # noqa: E402

from kgrag.config import Config             # noqa: E402
from kgrag.data import load_questions       # noqa: E402
from kgrag.factory import build_everything  # noqa: E402

st.set_page_config(page_title="KG-RAG vs Vector RAG", layout="wide")
st.title("Knowledge-Graph RAG vs Vector RAG")
st.caption("Same documents, same LLM, same prompt, same word budget. Only the retrieval method differs.")

hops = st.sidebar.slider("Graph hops", 1, 3, 2)
budget = st.sidebar.slider("Context budget (words)", 40, 300, 120, step=10)


@st.cache_resource
def load(hops, budget):
    return build_everything(Config(hops=hops, token_budget=budget))


env = load(hops, budget)
samples = load_questions(Config().questions_path)
choice = st.selectbox("Pick an example question", [q["question"] for q in samples])
question = st.text_input("...or type your own", value=choice)
q = next((x for x in samples if x["question"] == question),
         {"id": "custom", "question": question, "answer": "?", "supporting_titles": []})

if st.button("Answer") or True:
    names = {"S1": "Vector RAG", "S2": "KG-RAG", "S3": "Hybrid"}
    cols = st.columns(3)
    for col, s in zip(cols, [s for s in env["systems"] if s.name[:2] in names]):
        out = s.run(q)
        with col:
            st.subheader(s.name)
            st.markdown(f"**Answer:** {out.answer}")
            st.caption(f"gold: {q['answer']}")
            st.text(out.retrieval.text or "(empty context)")
            facts = [i for i in out.retrieval.items if "relation" in i]
            if facts:
                dot = "digraph G { rankdir=LR; node [shape=box, style=rounded];\n"
                for f in facts:
                    dot += f'"{f["subject"]}" -> "{f["object"]}" [label="{f["relation"]}"];\n'
                st.graphviz_chart(dot + "}")
