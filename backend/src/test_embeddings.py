from rag.packages.embeddings.ollama import OllamaEmbedder

embedder = OllamaEmbedder(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

documents = [
    "Sam Bankman-Fried was convicted in connection with fraud involving the cryptocurrency exchange FTX.",
    "He was sentenced to 110 months in prison and ordered to pay $3.9 billion in restitution.",
    "Sam Bankman-Fried was sentenced to 110 months in prison and ordered to pay $3.9 billion in restitution in connection with the FTX fraud case.",
    "The NFL hosts Monday Night Football games featuring professional American football teams.",
]

query = "What happened in the FTX fraud trial?"

document_vectors = embedder.embed_documents(documents)
query_vector = embedder.embed_query(query)

scored_documents = []

for document, vector in zip(documents, document_vectors):
    score = sum(a * b for a, b in zip(query_vector, vector))
    scored_documents.append((score, document))

for score, document in sorted(scored_documents, reverse=True):
    print(f"{score:.4f}  {document}")

# for document, vector in zip(documents, document_vectors):
#     score = sum(a * b for a, b in zip(query_vector, vector))
#     print(f"{score:.4f}  {document}")

# phrases = [
#     "FTX trial",
#     "Sam Bankman-Fried fraud case",
#     "NFL Monday Night Football",
# ]

# vectors = embedder.embed_documents(phrases)

# def cosine(a, b):
#     return sum(x * y for x, y in zip(a, b))

# for i in range(len(phrases)):
#     for j in range(i + 1, len(phrases)):
#         print(
#             f"{phrases[i]} <-> {phrases[j]}: "
#             f"{cosine(vectors[i], vectors[j]):.4f}"
#         )

# print(embedder.info())