![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-enabled-brightgreen)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


# LangChain RAG Demo (FAISS + Local Embeddings)

A minimal Retrieval-Augmented Generation (RAG) demo using **LangChain**, **FAISS** (local vector store), and **HuggingFace embeddings**.  
Optional UI via **Streamlit**. Works with **OpenAI** or **Ollama** (local LLM) as the generator.

> Designed as a fast portfolio project to show orchestration skills (LangChain / LlamaIndex-equivalent) for production-oriented roles.

## Features
- Document ingestion: chunking + local embeddings (`sentence-transformers/all-MiniLM-L6-v2`).
- Vector store: FAISS saved locally under `./storage/faiss_index`.
- RAG pipeline: retrieve top-k chunks and answer with cited sources.
- Two modes:
  - **CLI**: `rag_query.py --question "..."`.
  - **Streamlit** app: `streamlit run src/app.py`.
- Pluggable LLM provider: **OpenAI** or **Ollama**.

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### 2) Configure LLM (pick one)
- **OpenAI**: set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=...` in `.env`.
- **Ollama** (local): set `LLM_PROVIDER=ollama` and (optionally) `OLLAMA_MODEL=llama3.1` (or similar). Install Ollama and pull a model, e.g. `ollama pull llama3.1`.

### 3) Ingest documents
```bash
python src/ingest.py --docs_dir data/sample_docs --persist_dir storage/faiss_index
```

### 4) Query (CLI)
```bash
python src/rag_query.py --persist_dir storage/faiss_index --question "What is RAG and why is chunking useful?"
```

### 5) Streamlit UI (optional)
```bash
streamlit run src/app.py
```

## Project Structure
```
langchain-rag-demo/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ .env.example
├─ data/
│  └─ sample_docs/
│     ├─ 1_intro.txt
│     ├─ 2_rag.txt
│     └─ 3_eval.txt
├─ storage/            # created after ingestion
└─ src/
   ├─ ingest.py
   ├─ rag_query.py
   ├─ app.py
   └─ utils.py
```

## Notes
- Uses **local embeddings** to avoid external dependency for retrieval.
- LLM can be swapped; defaults to OpenAI if `LLM_PROVIDER` is unset and an API key exists, otherwise tries Ollama.
- For a cloud vector DB (Pinecone/Qdrant), swap FAISS with the appropriate LangChain vector store in `ingest.py`/`rag_query.py`.

## License
MIT

---

## LlamaIndex Variant

We also include a `llamaindex_demo/` directory with a minimal LlamaIndex-based RAG pipeline.

### Run LlamaIndex Demo
```bash
python llamaindex_demo/basic_rag.py --question "What is RAG?"
```


---

## NEW: LlamaIndex + Pinecone/Qdrant Variants

This project now includes:
- **LlamaIndex demo** (local FAISS) with CLI and Streamlit.
- **Vector DB backends** for LangChain: **Pinecone** and **Qdrant**.

### LlamaIndex (local FAISS)
```bash
pip install -r requirements.txt
python llamaindex_demo/src/ingest.py --docs_dir data/sample_docs --persist_dir storage/faiss_index_llama
python llamaindex_demo/src/query.py --persist_dir storage/faiss_index_llama --question "What is RAG?"
# UI:
streamlit run llamaindex_demo/src/app.py
```

### LangChain + Pinecone
```bash
# Env (.env):
# PINECONE_API_KEY=...
# PINECONE_INDEX=rag-demo-index
python src/ingest_pinecone.py --docs_dir data/sample_docs --index_name $PINECONE_INDEX
python src/rag_query_pinecone.py --index_name $PINECONE_INDEX --question "Explain chunking"
```

### LangChain + Qdrant
```bash
# Env (.env):
# QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=  # if cloud/secured
# QDRANT_COLLECTION=rag_demo
python src/ingest_qdrant.py --docs_dir data/sample_docs --collection $QDRANT_COLLECTION
python src/rag_query_qdrant.py --collection $QDRANT_COLLECTION --question "How to evaluate RAG?"
```


## Why this matters for production
- **Reproducibility**: scripted ingestion, deterministic chunking, and versioned embeddings.
- **Observability**: JSONL **metrics logging** (latency, token estimate, source count) + `reports/eval_report.md`.
- **Swap-in backends**: FAISS local for dev; **Pinecone/Qdrant** for scale without code rewrites.
- **Portability**: **Dockerfile** + `Makefile` targets; easy to run locally or in CI.
