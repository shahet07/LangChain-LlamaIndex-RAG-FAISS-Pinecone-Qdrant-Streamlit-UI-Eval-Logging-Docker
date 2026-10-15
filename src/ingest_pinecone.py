import argparse, os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

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
    parser.add_argument("--index_name", required=True)
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    args = parser.parse_args()

    load_dotenv()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # Ensure index exists (for demo: 384 dims for MiniLM)
    if args.index_name not in [i["name"] for i in pc.list_indexes()]:
        pc.create_index(name=args.index_name, dimension=384, metric="cosine")

    docs = load_docs(args.docs_dir)
    splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    _ = PineconeVectorStore.from_documents(chunks, embedding=embeddings, index_name=args.index_name)
    print(f"[PINECONE] Upserted {len(chunks)} chunks into index '{args.index_name}'")

if __name__ == "__main__":
    main()
