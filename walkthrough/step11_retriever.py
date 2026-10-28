"""Step 11: retrievers.

A retriever answers one question: "given this text, which Documents are relevant?"

    retriever.invoke("question")  →  list[Document]

`vector_store.as_retriever()` wraps our Postgres vector store in that interface. Why bother,
when the store can already search? Because a retriever is a *Runnable*, so it can sit inside a
chain with `|` (step 12), and every retrieval trick in LangChain (multi-query, reranking, hybrid)
is just another retriever with the same interface.

search_type picks *how* to choose results:
- "similarity": the k closest chunks, even if they all come from one article
- "mmr" (maximal marginal relevance): close to the question AND different from each other,
  so the results cover more articles. Useful when an answer needs facts from several sources.

Run:  uv run walkthrough/step11_retriever.py   (Postgres from step 10 must be running)
"""

from step10_postgres import vector_store

similarity = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
mmr = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 30})  # pick 5 out of the 30 closest

question = "Why is the FTC suing Amazon?"

for name, retriever in [("similarity", similarity), ("mmr", mmr)]:
    documents = retriever.invoke(question)
    print(f"\n{name}: {len(documents)} documents from {len({d.metadata['url'] for d in documents})} articles")
    for document in documents:
        print(f"  {document.metadata['title'][:70]}")

# A retriever is a Runnable, so batch() works too: several questions in one call.
questions = ["Where can I get discounts on Kindle and Echo devices?", "Who won the Raiders vs Lions game?"]
for q, documents in zip(questions, similarity.batch(questions)):
    print(f"\n{q}\n  → {documents[0].metadata['title'][:70]}")
