# Step 9: in-memory vector store

Notes for [`step09_vector_store.py`](step09_vector_store.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ InMemoryVectorStore | A Python dict + numpy | Demos, tests | Lost on exit; checks every vector |
| FAISS | Fast in-memory search library, can save to disk | Large local collections without a server | Its LangChain integration is in sunset langchain-community; weak metadata filtering |
| Chroma (langchain-chroma) | Embedded database with persistence | Local apps and prototypes | Single machine |
| pgvector (step 10) | Vectors inside Postgres | You already run Postgres; you want SQL + filters | Needs index tuning at very large scale |
| Qdrant / Weaviate / Milvus | Dedicated vector databases | Large scale, advanced filtering and hybrid search | Another service to run |
| Pinecone (managed) | Hosted vector database | No servers to manage | Cost; data is hosted elsewhere |

## This step's code

**When to use**

- Demos, notebooks and small datasets that can be re-embedded on every run.

**Limitations**

- Everything is lost when the program exits, so every run re-embeds all chunks.
- Needs numpy, which langchain-core doesn't install.
- Searches by comparing the question with every stored vector: slow for large collections.
- Subclassing embeddings is fragile: OllamaEmbeddings.embed_query calls self.embed_documents.
