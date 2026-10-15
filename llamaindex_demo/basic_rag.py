import argparse
import os
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama

def make_llm():
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY not set")
        return OpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    elif provider == "ollama":
        return Ollama(model=os.getenv("OLLAMA_MODEL", "llama3.1"), request_timeout=60.0)
    else:
        # fallback: try OpenAI if key is present, else Ollama
        if os.getenv("OPENAI_API_KEY"):
            return OpenAI(model="gpt-4o-mini", temperature=0)
        return Ollama(model="llama3.1", request_timeout=60.0)

def main():
    parser = argparse.ArgumentParser(description="Minimal RAG with LlamaIndex")
    parser.add_argument("--docs_dir", type=str, default="data/sample_docs")
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()

    load_dotenv()

    # load documents
    docs = SimpleDirectoryReader(args.docs_dir).load_data()

    # split
    parser = SimpleNodeParser.from_defaults(chunk_size=1000, chunk_overlap=150)
    nodes = parser.get_nodes_from_documents(docs)

    # embed + index
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    # query
    llm = make_llm()
    query_engine = index.as_query_engine(similarity_top_k=4, llm=llm)
    resp = query_engine.query(args.question)

    print("=== Answer ===")
    print(resp.response)
    print("\n=== Sources ===")
    if resp.source_nodes:
        for n in resp.source_nodes:
            print("-", n.node.metadata.get("file_path", "unknown"))

if __name__ == "__main__":
    main()
