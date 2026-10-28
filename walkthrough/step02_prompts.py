"""Step 2: prompt templates.

In step 1 we wrote the messages by hand. A *prompt template* is a reusable recipe for those
messages, with {placeholders} that are filled in at run time. In RAG, the same template will be
used for every question; only the {question} (and later the retrieved {context}) changes.

    template  +  variables   ──invoke──►  prompt value (the finished messages)  ──►  model

Run:  uv run walkthrough/step02_prompts.py
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

llm = ChatOllama(model="gemma4:12b", temperature=0, reasoning=False)

# (role, text) pairs: "system" → SystemMessage, "human" → HumanMessage.
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a concise assistant. Answer in one sentence, for a {audience}."),
        ("human", "{question}"),
    ]
)

# LangChain finds the placeholders for you.
print("Variables the template needs:", prompt.input_variables)

# 1) Fill the template: the result is the same kind of message list we built by hand in step 1.
prompt_value = prompt.invoke({"audience": "10-year-old", "question": "What is an embedding?"})
print("\nFilled-in messages:")
for message in prompt_value.to_messages():
    print(f"  {type(message).__name__}: {message.content}")

# 2) Send the filled prompt to the model.
print("\nReply:", llm.invoke(prompt_value).content)

# 3) Reuse the same template with different values: no copy-pasting of prompt text.
again = prompt.invoke({"audience": "software engineer", "question": "What is an embedding?"})
print("\nSame template, different audience:", llm.invoke(again).content)
