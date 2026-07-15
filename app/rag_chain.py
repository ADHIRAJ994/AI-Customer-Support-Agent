"""
Core RAG logic: format retrieved chunks into a numbered context block,
prompt the LLM to answer using bracketed citations like [1], [2], and
return both the answer text and structured Citation objects that the
frontend uses for source highlighting.
"""
import re
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

from app.llm import get_llm
from app.models import Citation

SYSTEM_PROMPT = """You are a helpful customer support assistant. Answer the \
user's question using ONLY the numbered context excerpts below. \
Every factual claim must include a bracketed citation like [1] or [1][2] \
referring to the excerpt number it came from. \
If the answer is not contained in the context, say you don't have that \
information in the knowledge base and suggest the user contact human support. \
Keep answers concise and directly useful. Do not invent citation numbers \
that are not in the context."""


def _format_context(docs: list[Document]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        lines.append(f"[{i}] (source: {source}, page: {page})\n{doc.page_content}")
    return "\n\n".join(lines)


def _build_history_messages(chat_history: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for turn in chat_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


def generate_answer(
    question: str, docs: list[Document], chat_history: list[dict]
) -> tuple[str, list[Citation]]:
    context = _format_context(docs)
    llm = get_llm()

    messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    messages.extend(_build_history_messages(chat_history))
    messages.append(
        HumanMessage(
            content=f"Context excerpts:\n\n{context}\n\nQuestion: {question}"
        )
    )

    response = llm.invoke(messages)
    answer_text = response.content if isinstance(response.content, str) else str(response.content)

    # Only surface citations the model actually referenced.
    cited_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer_text)})
    citations: list[Citation] = []
    for n in cited_numbers:
        if 1 <= n <= len(docs):
            doc = docs[n - 1]
            snippet = doc.page_content[:280].strip()
            citations.append(
                Citation(
                    id=n,
                    source=doc.metadata.get("source", "unknown"),
                    page=doc.metadata.get("page"),
                    snippet=snippet + ("..." if len(doc.page_content) > 280 else ""),
                )
            )
    return answer_text, citations