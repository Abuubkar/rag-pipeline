"""Step 20: follow-up questions (rewrite the question before searching).

Step 19 searched for the literal words "What did TechCrunch report about him?". The search engine
doesn't know who "him" is, so it found unrelated TechCrunch articles. The fix: before searching,
the LLM reads the history and rewrites the question so it stands on its own:

    history:   Who is facing a criminal trial for fraud …?  →  Sam Bankman-Fried …
    question:  What did TechCrunch report about him?
    rewritten: What did TechCrunch report about Sam Bankman-Fried?   ← this is what gets searched

The answer prompt still gets the original question plus the history, as in step 19.
(LangChain's create_history_aware_retriever did this for you; it now lives in the legacy
langchain-classic package, and inside it is the same small branch we write here.)

Run:  uv run walkthrough/step20_follow_up.py   (Postgres from step 10 and Ollama must be running)
"""

from operator import itemgetter

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from step13_citations import format_docs, llm, retriever
from step19_chat_history import prompt

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest question so it can be understood without the conversation above: "
            "replace words like 'him', 'it' or 'that' with what they refer to. Don't answer the question. "
            "If it is already complete, return it unchanged. Reply with the question only.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
rewriter = rewrite_prompt | llm | StrOutputParser()


def search_query(inputs: dict) -> str:
    if not inputs["history"]:  # first question: nothing to resolve, so skip the LLM call
        return inputs["question"]
    return rewriter.invoke(inputs)


# Step 19's chain with one new key: search with the rewritten question instead of the original.
chain = (
    RunnablePassthrough.assign(search_query=search_query)
    .assign(documents=itemgetter("search_query") | retriever)
    .assign(context=lambda inputs: format_docs(inputs["documents"]))
    .assign(answer=prompt | llm | StrOutputParser())
)

if __name__ == "__main__":
    history: list[BaseMessage] = []

    questions = [
        "Who is facing a criminal trial for fraud, as reported by The Verge?",
        "What did TechCrunch report about him?",
        "What was my first question?",
    ]

    for question in questions:
        result = chain.invoke({"question": question, "history": history})
        history += [HumanMessage(question), AIMessage(result["answer"])]

        print(f"\nQ: {question}")
        print(f"  searched for: {result['search_query']!r}")
        for document in result["documents"]:
            print(f"    {document.metadata['source'][:20]:20}  {document.metadata['title'][:60]}")
        print(f"A: {result['answer']}")
