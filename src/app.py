import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from utils import make_llm

load_dotenv()

st.set_page_config(page_title="LangChain RAG Demo", layout="wide")
st.title("LangChain RAG Demo (FAISS + Local Embeddings)")

persist_dir = st.text_input("Persist directory", "storage/faiss_index")
k = st.slider("Top-k", min_value=1, max_value=10, value=4)

col1, col2 = st.columns(2)

with col1:
    st.header("Build / Update Index")
    uploaded_files = st.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)
    if st.button("Ingest Uploaded Files"):
        if not uploaded_files:
            st.warning("Please upload at least one .txt file.")
        else:
            os.makedirs(persist_dir, exist_ok=True)
            temp_dir = ".streamlit_uploads"
            os.makedirs(temp_dir, exist_ok=True)

            docs = []
            for f in uploaded_files:
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as out:
                    out.write(f.read())
                docs.append(TextLoader(path, encoding="utf-8").load()[0])

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = splitter.split_documents(docs)
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            if os.path.exists(os.path.join(persist_dir, "index.faiss")):
                vdb = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
                vdb.add_documents(chunks)
            else:
                vdb = FAISS.from_documents(chunks, embeddings)
            vdb.save_local(persist_dir)
            st.success(f"Ingested {len(chunks)} chunks. Index saved to {persist_dir}.")

with col2:
    st.header("Query")
    question = st.text_area("Ask a question")
    if st.button("Run RAG"):
        if not os.path.exists(os.path.join(persist_dir, "index.faiss")):
            st.error("Index not found. Please ingest documents first.")
        else:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vdb = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
            retriever = vdb.as_retriever(search_kwargs={"k": k})
            llm = make_llm()

            from langchain.chains import RetrievalQA
            from langchain.prompts import PromptTemplate

            RAG_PROMPT = PromptTemplate.from_template(
                """Use the context to answer the user's question. If unknown, say you don't know.
Cite your sources by file name.

Question: {question}
Context:
{context}

Answer:"""
            )

            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": RAG_PROMPT},
            )
            resp = chain.invoke({"query": question})
            st.subheader("Answer")
            st.write(resp.get("result", ""))

            src_docs = resp.get("source_documents", [])
            if src_docs:
                st.subheader("Sources")
                for d in src_docs:
                    st.code(d.metadata.get("source", "unknown"))
