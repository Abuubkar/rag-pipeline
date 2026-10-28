"""Step 12: the first full RAG chain.

Everything so far, joined into one Runnable:

    question ─┬─► retriever ─► format_docs ─► {context} ─┐
              └─────────────────────────────► {question} ─┴─► prompt ─► llm ─► parser ─► answer

The dict at the start of the chain runs both branches on the same input (the question):
one fetches and formats the chunks, the other passes the question through unchanged.

Run:  uv run walkthrough/step12_rag_chain.py   (Postgres from step 10 and Ollama must be running)
"""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from step10_postgres import vector_store

retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the news excerpts below. "
            "If they don't contain the answer, reply 'Insufficient information.'\n\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(documents: list[Document]) -> str:
    # Turn the retrieved chunks into one block of text for the prompt, with their source.
    return "\n\n".join(f"[{d.metadata['title']} — {d.metadata['source']}]\n{d.page_content}" for d in documents)


# its same as writing: chain = StrOutputParser(llm(prompt({"content": ...})))
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = [
    "Why is the FTC suing Amazon?",
    # From the MultiHop-RAG question set: needs facts from several articles.
    "Who is the individual associated with the cryptocurrency industry facing a criminal trial "
    "on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?",
    # Not covered by the news corpus: should refuse.
    "What is the capital of Australia's best-selling pizza brand?",
]

for question in questions:
    print(f"\nQ: {question}")
    print(f"A: {chain.invoke(question)}")
