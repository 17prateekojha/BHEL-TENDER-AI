import os
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("RAG_DOCS_DIR", str(ROOT / "docs"))
os.environ.setdefault("RAG_INDEX_DIR", str(ROOT / ".rag_index"))

try:
    secrets = dict(st.secrets)
except st.errors.StreamlitSecretNotFoundError:
    secrets = {}

for key in ("ANTHROPIC_API_KEY", "GROQ_API_KEY", "RAG_PROVIDER", "RAG_LLM_MODEL"):
    if value := secrets.get(key):
        os.environ[key] = str(value)

import rag

st.set_page_config(
    page_title="BHEL Tender AI",
    page_icon=str(ROOT / "assets" / "bhel-logo.png"),
    layout="centered",
)

st.markdown(
    """
    <style>
    div[data-testid="stLogo"] {
        transform: scale(1.35);
        transform-origin: left center;
    }
    div[data-testid="stLogo"] img {
        width: 120px !important;
        height: auto !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.logo(str(ROOT / "assets" / "bhel-logo.png"), size="large")

SUGGESTIONS = [
    "Summarize the tender scope",
    "What are the key submission requirements?",
    "List the major safety requirements",
]


def show_sources(sources: list[dict]) -> None:
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for source in sources:
                st.write(f"{source['file']} · {source['loc']}")


def show_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["body"])
        show_sources(message.get("sources", []))


st.title("Tender assistant")
st.caption("Bharat Heavy Electricals Limited · Answers grounded in tender documents")

with st.sidebar:
    st.subheader("Tender documents")
    files = rag._files()
    st.caption(f"{len(files)} supported files in the document set")
    if st.button("Re-index documents", icon=":material/refresh:", width="stretch"):
        with st.spinner("Building the document index..."):
            result = rag.build_index()
        st.success(f"Indexed {result['files']} files into {result['chunks']} passages.")
    st.caption("Answers cite the source file and page or line location.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    show_message(message)

suggestion = None
if not st.session_state.messages:
    suggestion = st.pills(
        "Start with a question",
        SUGGESTIONS,
        label_visibility="collapsed",
        key="suggested_question",
    )

prompt = st.chat_input("Ask about the tender documents", submit_mode="disable")
prompt = prompt or suggestion

if prompt:
    st.session_state.messages.append({"role": "user", "body": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching tender documents..."):
            try:
                result = rag.answer(prompt)
            except rag.ProviderUnavailableError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"The question could not be answered: {exc}")
                st.stop()

        if result["refused"]:
            st.warning(result["body"])
        else:
            st.markdown(result["body"])
        show_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "body": result["body"],
            "sources": result["sources"],
            "refused": result["refused"],
        }
    )