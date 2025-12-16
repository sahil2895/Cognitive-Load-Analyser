import os
import streamlit as st
import re
from cognitive_load import compute_cli, explain_cli_result
from cognitive_rewriter import rewrite_basic_simplify

MODEL_NAME = "gemini-2.5-flash"

st.set_page_config(page_title="Cognitive Load–Aware Rewriter ", layout="centered")

st.title("Cognitive Load–Aware Rewriter ")

@st.cache_resource
def load_spacy():
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.max_length = 2_000_000
    return nlp

nlp = load_spacy()

with st.form("main"):
    text = st.text_area("Paste a paragraph (learning material):", height=220)
    col1, col2 = st.columns([1,1])
    with col1:
        target = st.selectbox("Target reading level (affects style):", ["Beginner","Intermediate","Advanced"], index=1)
    with col2:
        rewrite_btn = st.form_submit_button("Compute CLI")
st.write("---")

if text.strip():
    with st.spinner("Computing Cognitive Load Index..."):
        cli_data = compute_cli(nlp, text)
    st.subheader("Cognitive Load (CL2) — component breakdown")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Intrinsic", cli_data["intrinsic"]["intrinsic_score"])
    col_b.metric("Extraneous", cli_data["extraneous"]["extraneous_score"])
    col_c.metric("Germane", cli_data["germane"]["germane_score"])
    st.write(f"**Overall CLI:** {cli_data['cli']}  —  **{cli_data['label']}**")

    st.markdown("**Interpretation / Explanation**")
    for line in explain_cli_result(cli_data):
        st.write("- " + line)

    with st.expander("Show detailed diagnostics"):
        st.write("Intrinsic details:", cli_data["intrinsic"])
        st.write("Extraneous details:", cli_data["extraneous"])
        st.write("Germane details:", cli_data["germane"])

    st.markdown("---")
    st.subheader("Rewrite (R1: Basic Simplification)")
    st.write("Press the button to request Gemini to produce a simplified version (preserve main meaning).")
    if st.button("Rewrite to Reduce Cognitive Load"):
        if not os.getenv("GEMINI_API_KEY"):
            st.error("GEMINI_API_KEY not set. Set it in your terminal and restart Streamlit.")
        else:
            with st.spinner("Requesting rewrite from Gemini (basic simplification)..."):
                try:
                    rewritten = rewrite_basic_simplify(text, target_level=target, model_name=MODEL_NAME)
                except Exception as e:
                    st.error(f"Rewrite failed: {e}")
                    rewritten = ""
            # Side-by-side view
            st.markdown("### Before vs After")
            left, right = st.columns(2)
            with left:
                st.markdown("**Original**")
                st.write(text)
            with right:
                st.markdown("**Rewritten (Simplified)**")
                st.write(rewritten or "_(no output)_")
else:
    st.info("Paste a paragraph above and click 'Compute CLI' to analyze.")

st.caption("Prototype notes: CL2 = show Intrinsic/Extraneous/Germane + overall CLI. Rewriting uses Gemini 1.5 Flash and a simplification prompt (R1). For evaluation, collect human ratings and compare pre/post comprehension.")
