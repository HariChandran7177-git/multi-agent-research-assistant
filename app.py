import streamlit as st
from main import run_pipeline

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔬")

st.title("🔬 Multi-Agent Research Assistant")
st.caption("Planner → Researcher → Retriever → Critic → Reporter, powered by LangGraph")

query = st.text_input("Enter your research question:", placeholder="e.g. What are the latest trends in agentic AI?")

if st.button("Research", type="primary"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Running the multi-agent pipeline... this takes 30-60 seconds"):
            try:
                report = run_pipeline(query)
                st.success("Done!")
                st.markdown("### Final Report")
                st.markdown(report)
            except Exception as e:
                st.error(f"Something went wrong: {e}")