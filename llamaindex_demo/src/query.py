import argparse, os
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Settings

from llama_utils import get_llm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="storage/faiss_index_llama")
    parser.add_argument("--question", required=True)
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()

    load_dotenv()

    vector_store = FaissVectorStore.from_persist_dir(args.persist_dir)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store, storage_context=storage_context)
    Settings.llm = get_llm()

    query_engine = index.as_query_engine(similarity_top_k=args.k)
    resp = query_engine.query(args.question)
    print("\n=== Answer ===")
    print(str(resp))
    # Show sources if available
    try:
        nodes = resp.source_nodes
        if nodes:
            print("\n=== Sources ===")
            for n in nodes:
                md = n.metadata or {}
                print(md.get("file_name") or md.get("source") or "unknown")
    except Exception:
        pass

if __name__ == "__main__":
    main()
