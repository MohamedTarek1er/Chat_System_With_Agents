from pathlib import Path
import streamlit as st
from core.config import UPLOAD_DIR
from ingestion.loader import load_document
from ingestion.ingestion import ingest_documents
from ingestion.vector_store import delete_document, clear_vector_store
from graph.runner import run_pipeline

st.set_page_config(
    page_title="Multi-Agent Document QA System",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Multi-Agent RAG System")
st.caption("Powered by LangGraph, ChromaDB, and Multi-Agent Collaboration (Retriever, Analyst, Answer).")

# Sidebar - Document Ingestion & Settings
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, or TXT documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Documents", use_container_width=True):
        total_chunks = 0
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

        with st.spinner("Processing and ingesting documents..."):
            for uploaded_file in uploaded_files:
                safe_name = Path(uploaded_file.name).name
                file_path = UPLOAD_DIR / safe_name

                try:
                    file_path.write_bytes(uploaded_file.getbuffer())
                    docs = load_document(str(file_path))

                    # Ensure consistent doc_name metadata across all loaders
                    for d in docs:
                        d.metadata["doc_name"] = safe_name

                    num_chunks = ingest_documents(docs)
                    total_chunks += num_chunks
                    st.success(f"Ingested '{safe_name}' ({num_chunks} chunks)")
                except Exception as e:
                    st.error(f"Error processing {safe_name}: {e}")

        st.info(f"Total chunks indexed: {total_chunks}")

    # Delete Ingestion / Manage Existing Documents
    st.markdown("---")
    st.subheader("🗑️ Delete Ingestion")

    existing_files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()] if UPLOAD_DIR.exists() else []

    if existing_files:
        selected_file = st.selectbox(
            "Select document to delete",
            options=existing_files,
            key="delete_doc_select",
        )
        if st.button("Delete Selected Document", use_container_width=True):
            file_to_del = UPLOAD_DIR / selected_file
            delete_document(str(file_to_del.resolve()))
            delete_document(selected_file)
            if file_to_del.exists():
                file_to_del.unlink()
            st.success(f"Deleted '{selected_file}' from vector store and storage.")
            st.rerun()

    if st.button("Delete All Ingestion Data", use_container_width=True):
        clear_vector_store()
        if UPLOAD_DIR.exists():
            for f in UPLOAD_DIR.iterdir():
                if f.is_file():
                    f.unlink()
        st.success("All ingested documents and vector store data have been deleted.")
        st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Info")
    st.markdown(
        "- **Retriever:** Hybrid Semantic + BM25 + CrossEncoder Reranking\n"
        "- **Analyst:** Deep analysis & evidence sufficiency loop\n"
        "- **Answer:** Source citations & complete synthesis"
    )

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("📚 Retrieved Evidence & Sources"):
                for idx, src in enumerate(msg["sources"], start=1):
                    doc_name = src.get("doc_name", "Unknown document")
                    page = src.get("page")
                    page_info = f", Page {page}" if page else ""
                    st.markdown(f"**[{idx}] {doc_name}{page_info}**")
                    st.text(src.get("text", "").strip())

# Chat input
if user_query := st.chat_input("Ask a question about your documents..."):
    # Append and show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate answer with pipeline
    with st.chat_message("assistant"):
        with st.spinner("Agents are analyzing and synthesizing an answer..."):
            try:
                # Format previous history for the retriever
                chat_history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                result = run_pipeline(user_query, chat_history=chat_history)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                evidence_complete = result.get("evidence_complete", True)

                st.markdown(answer)

                if not evidence_complete:
                    st.warning("⚠️ Note: Maximum evidence-gathering loops reached. The answer may be based on partial evidence.")

                if sources:
                    with st.expander("📚 Retrieved Evidence & Sources"):
                        for idx, src in enumerate(sources, start=1):
                            doc_name = src.get("doc_name", "Unknown document")
                            page = src.get("page")
                            page_info = f", Page {page}" if page else ""
                            st.markdown(f"**[{idx}] {doc_name}{page_info}**")
                            st.text(src.get("text", "").strip())

                # Save assistant response to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                st.error(f"Error executing pipeline: {e}")
