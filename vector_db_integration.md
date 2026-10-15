# Vector DB Integration Notes

This project defaults to **FAISS** for simplicity.

To swap in a **cloud vector database**, change `ingest.py` and `rag_query.py` as follows:

## Pinecone

```python
from langchain_community.vectorstores import Pinecone
import pinecone

pinecone.init(api_key="YOUR_KEY", environment="YOUR_ENV")
index_name = "rag-demo"

# during ingestion
vectordb = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
```

## Qdrant

```python
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")  # or Qdrant cloud URL + API key
vectordb = Qdrant.from_documents(chunks, embeddings, client=client, collection_name="rag_demo")
```

Both Pinecone and Qdrant integrate seamlessly with LangChain retrievers.
