# Step 8: embeddings

Notes for [`step08_embeddings.py`](step08_embeddings.py).

## Approaches

✅ = used in this step's code.

| Approach | How it works | Use it when | Watch out for |
|---|---|---|---|
| ✅ Local model (nomic-embed-text via Ollama) | 768-number vectors, computed on your machine | Privacy, free, offline | Below the best hosted models; needs task prefixes |
| Hosted (OpenAI text-embedding-3, Voyage, Cohere) | An API returns the vectors | Best quality, many languages | Cost, data leaves your machine; switching vendor means re-embedding everything |
| Other open models (BGE, E5, GTE via sentence-transformers) | Run locally, many sizes and languages | A specific language or domain | Each has its own prefixes or instructions |
| ✅ Asymmetric (query vs document prefixes) | Questions and passages are embedded differently | Short questions searching long passages (RAG) | Forgetting the prefixes silently lowers quality |
| Symmetric | Both sides embedded the same way | Comparing similar texts (duplicates, clustering) | Worse for question → passage search |
| Sparse vectors (BM25, SPLADE) | One weight per word instead of 768 dense numbers | Exact names and rare words | Misses synonyms (step 14) |

## This step's code

**When to use**

- Finding text by meaning, even when it shares no words with the question.

**Limitations**

- Weak on exact names, codes and rare words (keyword search, step 14).
- Documents and queries must use the same model and the right prefixes; OllamaEmbeddings adds none.
- Changing the model (or prefixes) means re-embedding every chunk.
- "Close in meaning" isn't "answers the question" (reranking, step 17).
