import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_documents(docs_dir: str):
    docs_dir = Path(docs_dir)
    docs = []
    for p in sorted(docs_dir.glob("**/*.txt")):
        docs.append(TextLoader(str(p), encoding="utf-8").load()[0])
    if not docs:
        raise ValueError(f"No .txt documents found under: {docs_dir}")
    return docs

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS with local embeddings.")
    parser.add_argument("--docs_dir", type=str, required=True, help="Directory of .txt files.")
    parser.add_argument("--persist_dir", type=str, default="storage/faiss_index", help="Directory to persist FAISS index.")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    args = parser.parse_args()

    load_dotenv()

    print(f"[INGEST] Loading documents from {args.docs_dir} ...")
    docs = load_documents(args.docs_dir)

    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"[INGEST] Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)

    os.makedirs(args.persist_dir, exist_ok=True)
    vectordb.save_local(args.persist_dir)
    print(f"[INGEST] Saved FAISS index to: {args.persist_dir}")

if __name__ == "__main__":
    main()
