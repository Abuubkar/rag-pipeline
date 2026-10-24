"""Step 5: structured output.

So far the model returned free text. Code can't reliably read free text ("Yes, I think it's
probably about sports…"). With structured output you describe the shape you want as a
Pydantic class, and `with_structured_output` makes the model return exactly that: an object
with typed fields your code can use in `if` statements.

How it works under the hood: LangChain turns the class into a JSON schema, sends it to the
model, then parses the JSON back into the class. *How* it sends the schema matters (see below).

We'll use this later to decide things *about* a question before answering it
(guardrails in step 22, query rewriting in step 16).

Run:  uv run step05_structured.py
"""

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# The shape we want back. Field descriptions are instructions to the model (if it gets to see
# them: see `method` below), so writing them well is prompt engineering for the schema.
class QuestionAnalysis(BaseModel):
    topic: Literal["technology", "business", "sports", "entertainment", "science", "health", "other"] = Field(
        description="business includes finance, crypto, companies and court cases about them"
    )
    needs_recent_news: bool = Field(
        description="True if the question is about a specific event, person, game, trial or result "
        "that is reported in the news. False only for timeless facts (science, definitions)."
    )
    search_query: str = Field(description="A short keyword query to find relevant news articles")


llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)
# method="json_schema" (the default for Ollama) only CONSTRAINS generation to the schema; the
# model never reads the field descriptions. It returned valid objects with wrong values
# (the FTX fraud trial → topic "entertainment", needs_recent_news False).
# method="function_calling" sends the schema as a tool definition, so the model reads the
# descriptions → "business", True.
# Lesson: structured output guarantees the FORMAT, not the CORRECTNESS.
structured_llm = llm.with_structured_output(QuestionAnalysis, method="function_calling")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Analyse the user's question for a news search system."),
        ("human", "{question}"),
    ]
)

# Still a chain of Runnables; only the last step changed. The output is now a QuestionAnalysis.
chain = prompt | structured_llm

questions = [
    "What was Sam Bankman-Fried found guilty of in the FTX trial?",
    "Who won the Monday Night Football game between the Raiders and the Lions?",
    "What is the boiling point of water?",
]

for question in questions:
    result = chain.invoke({"question": question})
    print(f"\nQ: {question}")
    print(f"   type: {type(result).__name__}")
    print(f"   topic={result.topic!r}  needs_recent_news={result.needs_recent_news}")
    print(f"   search_query={result.search_query!r}")

    # The point: code can now make decisions from the model's output.
    if not result.needs_recent_news:
        print("   → general knowledge; a RAG system could answer without searching")
