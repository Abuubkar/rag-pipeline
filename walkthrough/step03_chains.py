"""Step 3: chains with LCEL (LangChain Expression Language).

In step 2 we ran each piece by hand:   prompt.invoke(...)  →  llm.invoke(...)  →  reply.content
Every piece is a *Runnable*, so they can be joined with `|`. The output of one becomes the input
of the next, and the result is itself a Runnable with invoke / stream / batch.

    dict ──► prompt ──► messages ──► llm ──► AIMessage ──► parser ──► str

The prompt already has the RAG shape: {context} + {question}. For now we write the context
by hand; from step 12 a retriever will fill it automatically.

Run:  uv run walkthrough/step03_chains.py
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer ONLY from the context below. If the answer is not in it, reply "
            "'Insufficient information.'\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

# StrOutputParser turns the AIMessage into a plain string (it takes `.content` for us).
chain = prompt | llm | StrOutputParser()

print("The chain:", " | ".join(type(step).__name__ for step in chain.steps), "\n")

context = (
    "Sam Bankman-Fried founded the crypto exchange FTX. In October 2023 a New York jury "
    "found him guilty of seven counts of fraud and conspiracy."
)

# 1) One call runs all three steps. The input is the dict the prompt needs.
answer = chain.invoke({"context": context, "question": "What was Sam Bankman-Fried found guilty of?"})
print("Answer:", answer)
# A str, not an AIMessage: the parser did that. (LangChain v1 returns a str subclass
# called TextAccessor, kept for backward compatibility. It behaves exactly like str.)
print("Is a string:", isinstance(answer, str))

# 2) The context doesn't contain this answer, so the instructions should make it refuse.
print("\nNot in context:", chain.invoke({"context": context, "question": "Who won the 2023 World Series?"}))

# 3) A chain is a Runnable, so batch() works for free: both questions run in parallel.
print("\nBatch:")
questions = ["Which company did he found?", "When was the verdict?"]
for q, a in zip(questions, chain.batch([{"context": context, "question": q} for q in questions])):
    print(f"  {q} → {a}")
