"""Step 13: sources and citations.

Step 12's chain returned only the answer string, so you couldn't check where it came from.
Here the chain returns a dict that keeps everything:

    {"question": ..., "documents": [...], "context": "...", "answer": "..."}

RunnablePassthrough.assign(key=runnable) keeps its input dict and ADDS one key, computed by the
runnable from that dict. Chaining .assign() calls builds the dict up step by step.

The chunks are numbered [1]..[5] in the prompt and the model is asked to cite them, so each
claim in the answer can be traced back to an article.

Run:  uv run step13_citations.py
"""

from operator import itemgetter

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

from step10_postgres import vector_store

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using ONLY the numbered news excerpts below. "
            "Cite the excerpts you used with their numbers, like [1] or [2][4]. "
            "If they don't contain the answer, reply 'Insufficient information.'\n\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def format_docs(documents: list[Document]) -> str:
    return "\n\n".join(f"[{n}] {d.metadata['title']} — {d.metadata['source']}\n{d.page_content}"
                       for n, d in enumerate(documents, start=1))


chain = (
    RunnablePassthrough.assign(documents=itemgetter("question") | retriever)       # + documents
    .assign(context=lambda inputs: format_docs(inputs["documents"]))             # + context
    .assign(answer=prompt | llm | StrOutputParser())                             # + answer
)

# Runs only with `uv run step13_citations.py`, not when another step imports the chain.
if __name__ == "__main__":
    result = chain.invoke({"question": "Why is the FTC suing Amazon?"})

    print("Keys returned:", list(result))
    print(f"\nAnswer:\n{result['answer']}")
    print("\nSources:")
    for n, document in enumerate(result["documents"], start=1):
        print(f"  [{n}] {document.metadata['title'][:60]} — {document.metadata['source']}")
        print(f"      {document.metadata['url']}")
