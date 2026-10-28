"""Step 4: streaming.

`invoke` waits until the whole answer is generated, then returns it.
`stream` returns a generator: you get the answer piece by piece (a few characters, called
"chunks") as the model produces them. The total time is the same, but the user sees the first
words almost immediately. That's why every chat app streams.

Because every piece of the chain is a Runnable, streaming the *whole chain* just works:
the prompt runs once, the model streams, and StrOutputParser passes each chunk through as text.

Run:  uv run walkthrough/step04_streaming.py
"""

import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer in about 80 words."),
        ("human", "{question}"),
    ]
)
chain = prompt | llm | StrOutputParser()
question = {"question": "Explain what a vector database is and why RAG needs one."}

# 1) invoke: nothing is shown until the full answer is ready.
start = time.perf_counter()
answer = chain.invoke(question)
print(f"invoke → waited {time.perf_counter() - start:.1f}s, then got {len(answer)} characters at once\n")

# 2) stream: print each chunk as soon as it arrives.
start = time.perf_counter()
first_chunk_at = None
chunks = 0
for chunk in chain.stream(question):
    if first_chunk_at is None:
        first_chunk_at = time.perf_counter() - start
    chunks += 1
    print(chunk, end="", flush=True)  # flush=True: show it now, don't buffer
total = time.perf_counter() - start

print(f"\n\nstream → first words after {first_chunk_at:.1f}s, finished after {total:.1f}s, {chunks} chunks")
