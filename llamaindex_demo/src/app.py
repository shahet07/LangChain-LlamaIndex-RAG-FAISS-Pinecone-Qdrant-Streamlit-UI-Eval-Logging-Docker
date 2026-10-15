import streamlit as st
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_utils import get_llm

st.set_page_config(page_title="LlamaIndex RAG Demo", layout="wide")
st.title("LlamaIndex RAG Demo (FAISS + Local Embeddings)")

persist_dir = st.text_input("Persist directory", "storage/faiss_index_llama")
k = st.slider("Top-k", 1, 10, 4)

if st.button("Load Index"):
    try:
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        st.session_state.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, storage_context=storage_context
        )
        Settings.llm = get_llm()
        st.success("Index loaded.")
    except Exception as e:
        st.error(f"Failed to load index: {e}")

question = st.text_area("Ask a question")
if st.button("Query"):
    if "index" not in st.session_state:
        st.error("Load an index first.")
    else:
        qe = st.session_state.index.as_query_engine(similarity_top_k=k)
        resp = qe.query(question)
        st.subheader("Answer")
        st.write(str(resp))
        try:
            nodes = resp.source_nodes
            if nodes:
                st.subheader("Sources")
                for n in nodes:
                    md = n.metadata or {}
                    st.code(md.get("file_name") or md.get("source") or "unknown")
        except Exception:
            pass
