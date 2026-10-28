"""Every prompt the pipeline sends to the LLM, in one place."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# retrieval.py: split a question into (query, publication) searches. (walkthrough step 18)
PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You plan searches for a news search engine. Split the user's question into its separate parts "
            "and write one short search query for each part. If the question says which publication reported "
            "a part, set that search's source to the publication's name; otherwise set source to null.\n\n"
            "Publications: {sources}",
        ),
        ("human", "{question}"),
    ]
)

# reranking.py: score one excerpt from 0 to 10. (walkthrough step 17)
GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade how useful a news excerpt is for answering a question. Reply with a score from 0 to 10:\n"
            "10 = it directly answers the question or one part of it\n"
            "5 = same topic, but doesn't answer it\n"
            "0 = unrelated",
        ),
        ("human", "Question: {question}\n\nExcerpt:\n{excerpt}"),
    ]
)

# generation.py: answer from numbered excerpts, with the conversation so far. (walkthrough steps 13, 19)
ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer questions about the news using ONLY the numbered news excerpts below, and cite them like "
            "[1] or [2][4]. Questions about this conversation itself can be answered from the conversation. "
            "If neither contains the answer, reply 'Insufficient information.'\n\n{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)
