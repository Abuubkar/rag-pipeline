"""The chat model. (walkthrough step 1)"""

from functools import cache

from langchain_ollama import ChatOllama

from rag_langchain.config import settings


# @cache: created on first use, then shared. Nothing connects to Ollama at import time.
@cache
def chat_model() -> ChatOllama:
    return ChatOllama(model=settings.chat_model, temperature=0, reasoning=False)
