import os

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Baseline Demo", layout="wide")
st.title("RAG Baseline Demo")
st.caption("Обычный RAG без модуля противоречий")


def get_health() -> dict:
    response = requests.get(f"{API_URL}/health", timeout=30)
    response.raise_for_status()
    return response.json()


def ingest_directory(directory: str) -> dict:
    response = requests.post(
        f"{API_URL}/ingest",
        json={"directory": directory},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def upload_files(uploaded_files: list) -> dict:
    files = []
    for uploaded in uploaded_files:
        files.append(
            (
                "files",
                (
                    uploaded.name,
                    uploaded.getvalue(),
                    uploaded.type or "application/octet-stream",
                ),
            )
        )

    response = requests.post(f"{API_URL}/ingest/files", files=files, timeout=300)
    response.raise_for_status()
    return response.json()


def ask_question(question: str, limit: int) -> dict:
    response = requests.post(
        f"{API_URL}/ask",
        json={"question": question, "top_k": limit},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


left, right = st.columns([1, 2])

with left:
    st.subheader("Indexing")

    try:
        health = get_health()
        st.success(
            f"API is online. Indexed chunks: {health['indexed_chunks']}. "
            f"Groq enabled: {health['groq_enabled']}"
        )
    except Exception as error:
        st.error(f"API unavailable: {error}")

    directory = st.text_input("Default directory seen by local backend", "../back/data/raw")
    if st.button("Index default directory", use_container_width=True):
        try:
            result = ingest_directory(directory)
            st.json(result)
        except Exception as error:
            st.error(str(error))

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "html", "htm", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Upload and index files", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one file.")
        else:
            try:
                result = upload_files(uploaded_files)
                st.json(result)
            except Exception as error:
                st.error(str(error))

with right:
    st.subheader("Ask")
    top_k = st.slider("Top-k", min_value=1, max_value=10, value=5)
    question = st.text_area(
        "Question",
        placeholder="Например: Какая версия документа считается актуальной?",
        height=120,
    )

    if st.button("Ask", type="primary", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            try:
                answer = ask_question(question, top_k)
                st.markdown("### Answer")
                st.write(answer["answer"])
                st.markdown("### Retrieved chunks")
                for hit in answer["hits"]:
                    with st.expander(
                        f"{hit['source_path']} | score={hit['score']:.3f}",
                        expanded=False,
                    ):
                        st.write(hit["text"])
                        st.json(hit["metadata"])
            except Exception as error:
                st.error(str(error))
