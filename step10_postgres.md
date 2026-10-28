# Step 10: persistent vector store (Postgres + pgvector)

Notes for [`step10_postgres.py`](step10_postgres.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ No index (exact scan) | Compares the question with every row | Thousands of rows (we have 8,448); exact results | Gets slower as rows grow |
| HNSW index | A graph of neighbours; approximate search | Most production cases: fast queries | More memory; slower inserts and build |
| IVFFlat index | Groups vectors into clusters; searches the nearest clusters | Large, mostly static data with less memory | Build after loading data; recall depends on lists / probes |
| ✅ Cosine distance | Angle between vectors | Text embeddings (the usual choice) | Ignores vector length |
| Inner product | Dot product | Normalised vectors: same order as cosine, a bit faster | Wrong results if vectors aren't normalised |
| Euclidean (L2) | Straight-line distance | Embeddings where vector length carries meaning | Rarely right for text |
| ✅ PGVectorStore (vs deprecated PGVector) | Table per store; hybrid search built in | New projects | Rough edges: no create-if-missing, shared-config bug (step 14) |

## This step's code

**When to use**

- Vectors must survive restarts; embed once, search many times.
- You want SQL, metadata filters and keyword search in the same database.

**Limitations**

- Sync methods call the async embedding methods, so embedding subclasses must override both.
- init_vectorstore_table has no "create if missing"; overwrite_existing=True drops all vectors.
- Scores are cosine distances (lower = better), unlike InMemoryVectorStore's similarities.
- No vector index is set up here: every search scans all 8,448 rows (fine now; add HNSW for millions).
