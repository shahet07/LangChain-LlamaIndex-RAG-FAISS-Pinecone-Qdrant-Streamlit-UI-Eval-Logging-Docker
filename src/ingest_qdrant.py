import argparse, os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

def load_docs(docs_dir):
    out = []
    for root, _, files in os.walk(docs_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                path = os.path.join(root, fn)
                out.append(TextLoader(path, encoding="utf-8").load()[0])
    if not out:
        raise ValueError("No .txt docs found")
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    args = parser.parse_args()

    load_dotenv()

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY") or None
    client = QdrantClient(url=url, api_key=api_key)

    docs = load_docs(args.docs_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _ = Qdrant.from_documents(
        chunks,
        embedding=embeddings,
        location=url,
        prefer_grpc=False,
        api_key=api_key,
        collection_name=args.collection,
    )
    print(f"[QDRANT] Upserted {len(chunks)} chunks into collection '{args.collection}'")

if __name__ == "__main__":
    main()
