"""
LangGraph orchestration of the support agent.

Flow:
  START -> rewrite_question -> retrieve -> generate -> END

- rewrite_question: turns a follow-up like "what about refunds on that?"
  into a standalone query using chat history (history-aware retrieval).
- retrieve: semantic search against FAISS.
- generate: calls the LLM with numbered context and produces a cited answer.

Multi-turn state is persisted per thread_id via LangGraph's MemorySaver
checkpointer, so the graph itself is stateless between calls and the
caller only needs to pass a thread_id.
"""
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from app.vectorestore import get_retriever
from app.rag_chain import generate_answer
from app.llm import get_llm
from app.models import Citation


class ConversationState(TypedDict):
    question: str
    standalone_question: str
    chat_history: list[dict]  # [{"role": "user"|"assistant", "content": str}, ...]
    documents: list[Document]
    answer: str
    citations: list[dict]


REWRITE_PROMPT = """Given the conversation history and a follow-up question, \
rewrite the follow-up into a standalone question that contains all \
necessary context. If it is already standalone, return it unchanged. \
Return ONLY the rewritten question, nothing else."""


def rewrite_question_node(state: ConversationState) -> dict:
    if not state["chat_history"]:
        return {"standalone_question": state["question"]}

    history_text = "\n".join(
        f'{turn["role"]}: {turn["content"]}' for turn in state["chat_history"][-6:]
    )
    llm = get_llm(temperature=0)
    messages = [
        SystemMessage(content=REWRITE_PROMPT),
        HumanMessage(
            content=f"History:\n{history_text}\n\nFollow-up question: {state['question']}"
        ),
    ]
    result = llm.invoke(messages)
    rewritten = result.content if isinstance(result.content, str) else state["question"]
    return {"standalone_question": rewritten.strip() or state["question"]}


def retrieve_node(state: ConversationState) -> dict:
    retriever = get_retriever()
    docs = retriever.invoke(state["standalone_question"])
    return {"documents": docs}


def generate_node(state: ConversationState) -> dict:
    answer, citations = generate_answer(
        state["standalone_question"], state["documents"], state["chat_history"]
    )
    return {
        "answer": answer,
        "citations": [c.model_dump() for c in citations],
    }


def build_graph():
    workflow = StateGraph(ConversationState)
    workflow.add_node("rewrite_question", rewrite_question_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "rewrite_question")
    workflow.add_edge("rewrite_question", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)


_graph = build_graph()

# Per-thread chat history kept alongside the checkpointer for simplicity.
# (LangGraph's checkpointer persists full state per thread already; this
# dict just makes it easy to append/read the rolling history.)
_histories: dict[str, list[dict]] = {}


def run_turn(thread_id: str, question: str) -> tuple[str, list[Citation], list[dict]]:
    history = _histories.setdefault(thread_id, [])

    config = {"configurable": {"thread_id": thread_id}}
    result = _graph.invoke(
        {
            "question": question,
            "chat_history": history,
        },
        config=config,
    )

    answer = result["answer"]
    citations = [Citation(**c) for c in result["citations"]]

    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    _histories[thread_id] = history[-20:]  # cap history length

    return answer, citations, _histories[thread_id]


def reset_thread(thread_id: str) -> None:
    _histories.pop(thread_id, None)