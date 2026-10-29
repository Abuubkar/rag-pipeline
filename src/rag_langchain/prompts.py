"""Every prompt the pipeline sends to the LLM, in one place."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# retrieval.py: turn a follow-up ("what about him?") into a stand-alone question. (walkthrough step 20)
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
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
# Shows the search that found the excerpt, so a chunk answering one part of the question isn't scored 0.
GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You grade how useful a news excerpt is for answering a question. The question may have several parts; "
            "judge the excerpt by the part it answers best. Reply with a score from 0 to 10:\n"
            "10 = it directly answers the question or one part of it\n"
            "5 = same topic, but doesn't answer it\n"
            "0 = unrelated",
        ),
        ("human", "Question: {question}\nThe excerpt was found by searching for: {query}\n\nExcerpt:\n{excerpt}"),
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
