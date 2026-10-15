import argparse, os
from dotenv import load_dotenv

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from utils import make_llm
from logging_utils import timeit, append_jsonl
from token_utils import estimate_tokens

RAG_PROMPT = PromptTemplate.from_template(
    """Use the context to answer the user's question. If unknown, say you don't know.
Cite your sources by file name.

Question: {question}
Context:
{context}

Answer:"""
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    load_dotenv()

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vdb = Qdrant(
        client=QdrantClient(url=url, api_key=api_key),
        collection_name=args.collection,
        embeddings=embeddings,
    )
    retriever = vdb.as_retriever(search_kwargs={"k": args.k})

    chain = RetrievalQA.from_chain_type(
        llm=make_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )

    with timeit() as t:
    resp = chain.invoke({"query": args.question})
latency = t()

    print("\n=== Answer ===")
print(f"[METRICS] latency_sec={latency:.3f}")

    print(resp.get("result", ""))
    print("\n=== Sources ===")
    for d in resp.get("source_documents", []):
        print(d.metadata.get("source", "unknown"))

if __name__ == "__main__":
    main()

# --- Metrics JSONL logging ---
try:
    answer = resp.get('result','')
    srcs = [d.metadata.get('source','unknown') for d in resp.get('source_documents', [])]
    rec = {
        'backend': 'qdrant',
        'question': args.question,
        'latency_sec': round(latency,3),
        'answer_len_chars': len(answer),
        'answer_est_tokens': estimate_tokens(answer),
        'num_sources': len(srcs),
        'sources': srcs,
    }
    append_jsonl(rec, 'reports/rag_cli_metrics.jsonl')
except Exception:
    pass
