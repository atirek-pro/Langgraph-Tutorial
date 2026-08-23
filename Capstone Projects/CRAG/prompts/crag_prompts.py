from langchain_core.prompts import ChatPromptTemplate

query_rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a search query optimization assistant.

Rewrite the user's question into a concise,
fact-focused search query that will work well
with a web search engine.

Rules:
- Preserve the original intent.
- Do not answer the question.
- Do not add facts that are not present in the question.
- Remove unnecessary conversational wording.
- Keep important entities, names, dates, and technical terms.
- Return ONLY the rewritten search query.
""",
        ),
        (
            "human",
            "User question:\n{question}",
        ),
    ]
)

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. "
            "Answer ONLY using the provided "
            "refined context. "
            "Do not use outside knowledge. "
            "If the refined context is empty "
            "or insufficient, say exactly: "
            "'I don't know based on the provided books.'",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Refined context:\n"
            "{refined_context}",
        ),
    ]
)