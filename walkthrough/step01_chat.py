"""Step 1: call a local LLM through LangChain.

A *chat model* takes a list of messages and returns one new message.
- SystemMessage: instructions for how the model should behave
- HumanMessage:  what the user says
- AIMessage:     what the model replies (this is what `invoke` returns)

Run:  uv run walkthrough/step01_chat.py   (Ollama must be running)
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# ChatOllama talks to the Ollama server on this machine (http://localhost:11434 by default).
# temperature=0 → the most likely answer every time, which makes results repeatable.
# reasoning=False → gemma4 is a "thinking" model; by default it reasons privately before
#   answering, and those hidden tokens cost time. Try True to compare output_tokens.
llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)

messages = [
    SystemMessage("You are a concise assistant. Answer in at most two sentences."),
    HumanMessage("What is retrieval-augmented generation?"),
]

reply = llm.invoke(messages)  # returns an AIMessage

print("Reply:", reply.content)
print("Message type:", type(reply).__name__)
print("Token usage:", reply.usage_metadata)
