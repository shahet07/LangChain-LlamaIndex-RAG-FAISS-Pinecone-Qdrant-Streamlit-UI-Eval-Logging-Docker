import argparse
import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from utils import make_llm
from logging_utils import timeit, append_jsonl
from token_utils import estimate_tokens

RAG_PROMPT = PromptTemplate.from_template(
    """You are a helpful assistant. Use the provided context to answer the question.
If the answer is not in the context, say you don't know. Cite your sources by file name.

Question: {question}
Context:
{context}

Answer:"""
)

def load_db(persist_dir: str):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

def make_chain(vdb, k: int = 4):
    retriever = vdb.as_retriever(search_kwargs={"k": k})
    llm = make_llm()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    return chain

def sources_to_str(source_docs):
    names = []
    for d in source_docs:
        meta = d.metadata or {}
        name = os.path.basename(meta.get("source", "unknown"))
        names.append(name)
    # unique order-preserving
    seen = set()
    out = []
    for n in names:
        if n not in seen:
            out.append(n); seen.add(n)
    return ", ".join(out) if out else "N/A"

def main():
    parser = argparse.ArgumentParser(description="Query a FAISS-backed RAG index.")
    parser.add_argument("--persist_dir", type=str, default="storage/faiss_index")
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    load_dotenv()

    vdb = load_db(args.persist_dir)
    chain = make_chain(vdb, k=args.k)
    with timeit() as t:
    resp = chain.invoke({"query": args.question})
latency = t()


    answer = resp.get("result", "").strip()
    source_docs = resp.get("source_documents", [])
    src_str = sources_to_str(source_docs)
# Log JSONL metrics
record = {
    "question": args.question,
    "latency_sec": round(latency, 3),
    "answer_len_chars": len(answer),
    "answer_est_tokens": estimate_tokens(answer),
    "num_sources": len(source_docs),
    "sources": src_str.split(", ") if src_str != "N/A" else []
}
append_jsonl(record, "reports/rag_cli_metrics.jsonl")

    print("\n=== Answer ===")
print(f"[METRICS] latency_sec={latency:.3f}")

    print(answer)
    print("\n=== Sources ===")
    print(src_str)

if __name__ == "__main__":
    main()
