import argparse, os
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", required=True)
    parser.add_argument("--persist_dir", default="storage/faiss_index_llama")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    args = parser.parse_args()

    load_dotenv()

    docs = SimpleDirectoryReader(args.docs_dir, recursive=True).load_data()
    splitter = SentenceSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    Settings.node_parser = splitter
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Prepare FAISS store
    dim = 384  # all-MiniLM-L6-v2
    index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    vs_index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)

    os.makedirs(args.persist_dir, exist_ok=True)
    vector_store.save(args.persist_dir)
    print(f"[LLAMA] Saved FAISS index to {args.persist_dir}")

if __name__ == "__main__":
    main()
