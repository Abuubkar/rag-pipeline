"""Step 19: chat history (remembering earlier turns).

A chat model has no memory: every invoke() starts from zero (step 1). To hold a conversation, you
send the earlier turns again with every new question. History is just more messages in the prompt:

    System: Answer using ONLY the excerpts … {context}
    Human:  Who is on trial for fraud, as reported by The Verge?   ┐
    AI:     Sam Bankman-Fried … [1]                                ┘ ← MessagesPlaceholder("history")
    Human:  What did TechCrunch say about him?                       ← the new question

The history is a plain Python list: after each turn we append the question and the answer.
(LangChain's InMemoryChatMessageHistory and RunnableWithMessageHistory used to do this for you,
but both are deprecated in v1; the docs now point to LangGraph persistence. See ROADMAP.)

Run:  uv run walkthrough/step19_chat_history.py   (Postgres from step 10 and Ollama must be running)
"""

from operator import itemgetter

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from step13_citations import format_docs, llm, retriever

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer questions about the news using ONLY the numbered news excerpts below, and cite them like [1]. "
            "Questions about this conversation itself can be answered from the conversation. "
            "If neither contains the answer, reply 'Insufficient information.'\n\n{context}",
        ),
        MessagesPlaceholder("history"),  # the earlier turns are inserted here
        ("human", "{question}"),
    ]
)

# Step 13's chain; only the prompt changed. Retrieval still uses just the new question.
chain = (
    RunnablePassthrough.assign(documents=itemgetter("question") | retriever)
    .assign(context=lambda inputs: format_docs(inputs["documents"]))
    .assign(answer=prompt | llm | StrOutputParser())
)

history: list[BaseMessage] = []  # the whole "memory" of this conversation

questions = [
    "Who is facing a criminal trial for fraud, as reported by The Verge?",
    "What did TechCrunch report about him?",  # "him" only makes sense with the history
    "What was my first question?",            # answerable from the history alone
]

for question in questions:
    result = chain.invoke({"question": question, "history": history})
    history += [HumanMessage(question), AIMessage(result["answer"])]  # remember this turn

    print(f"\nQ: {question}")
    print(f"  searched for: {question!r}")
    for document in result["documents"]:
        print(f"    {document.metadata['source'][:20]:20}  {document.metadata['title'][:60]}")
    print(f"A: {result['answer']}")
    print(f"  (history now holds {len(history)} messages)")
