# Multi-Agent RAG Document QA System

A modular Retrieval-Augmented Generation (RAG) system built with LangGraph, ChromaDB, and collaborative specialized agents (Retriever, Analyst, and Answer).

---

## Project Structure

```text
Chat_System_With_Agents/
├── app.py                   # Streamlit web UI
├── requirements.txt         # Project dependencies
├── CONTRACT.md              # Pipeline contracts, data formats, and rules
├── .env                     # Environment variables (API keys, models, DB paths)
│
├── api/                     # (Planned) FastAPI REST API endpoints & route handlers
│
├── core/                    # Shared infrastructure
│   ├── __init__.py
│   ├── config.py            # Centralized settings (API keys, models, loop cap, DB)
│   ├── llm.py               # Shared ChatOpenAI client initialization
│   └── prompts.py           # All centralized agent and tool prompt templates
│
├── graph/                   # LangGraph orchestration
│   ├── __init__.py
│   ├── state.py             # State TypedDict, EvidenceItem schema, and converters
│   ├── workflow.py          # StateGraph assembly, nodes, loop counter, and router
│   └── runner.py            # Pipeline entry point: run_pipeline()
│
├── ingestion/               # Document processing & indexing pipeline
│   ├── __init__.py
│   ├── loader.py            # PDF, DOCX, and TXT loaders with metadata extraction
│   ├── splitter.py          # Recursive text splitting into chunks
│   ├── vector_store.py      # ChromaDB storage, similarity search, chunk IDs
│   └── ingestion.py         # End-to-end ingestion orchestrator (split -> hash -> index)
│
└── agents/                  # Domain-specific agents & their tools
    ├── retriever/           # Hybrid retriever agent
    │   ├── retriever_agent.py
    │   └── tools/           # Semantic search, BM25, query rewriter, reranker, filters
    ├── analyst/             # Evidence analysis & sufficiency agent
    │   ├── analyst_agent.py
    │   └── tools/           # Data analysis, document comparison, evidence request
    ├── answer/              # Answer synthesis & citation agent
    │   ├── answer_agent.py
    │   └── tools/           # Response generator, source formatter, citation formatter
    └── generator/           # (Planned) Content & report generation agent
```

---

## File Structure Changes & Refactoring Summary

The codebase was refactored from flat and scattered directories into a clean, modular architecture:

1. **Centralized `core/` Module**:
   - **`config.py`**: Consolidated all environment variables, model names, retrieval parameters (`RETRIEVER_CANDIDATES`, `RERANK_TOP_K`, `MAX_CHARS`), and orchestrator settings (`MAX_LOOPS`) in one place.
   - **`prompts.py`**: Extracted all hardcoded and inline LLM prompts into reusable, parameterized templates.
   - **`llm.py`**: Single source of truth for the LLM client, configured via `core.config`.

2. **Decoupled `graph/` Package**:
   - Deconstructed the previous monolithic orchestration logic into distinct modules:
     - `state.py`: Explicit `State` and `EvidenceItem` types matching `CONTRACT.md`.
     - `workflow.py`: Pure node functions, conditional routing (`after_analyst`), and loop-cap enforcement.
     - `runner.py`: Public pipeline entry point (`run_pipeline`) with cached compiled graph execution.

3. **Consolidated `ingestion/` Pipeline**:
   - Replaced the legacy `document_processing/` directory with `ingestion/`.
   - Unified document loading, chunk splitting, metadata enrichment (`doc_name`, `page_number`, `chunk_id`), and vector indexing into ChromaDB.

4. **Grouped `agents/` Hierarchy**:
   - Merged separate top-level folders (`retriever_tools/`, `analyst_tools/`, `answer_tools/`) into dedicated subpackages under `agents/`:
     - `agents/retriever/` (with `tools/`)
     - `agents/analyst/` (with `tools/`)
     - `agents/answer/` (with `tools/`)
   - Cleaned up import paths across all agents to use standardized package imports.

5. **Planned Additions**:
   - **`api/`**: Reserved for forthcoming FastAPI REST endpoints to expose the pipeline via API services.
   - **`agents/generator/`**: A planned Generator agent dedicated to structured draft and content generation.

---

> [!NOTE]
> **Future Refactoring / OOP Conversion**:  
> Several components and tools across the codebase are currently implemented as flat procedural functions. In future iterations, several parts of the codebase will need to be refactored and converted into object-oriented classes (e.g., dedicated loaders, retriever classes, agent base/service classes, and manager patterns) to improve extensibility, encapsulation, and state handling.
