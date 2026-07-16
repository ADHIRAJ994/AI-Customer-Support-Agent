"""
Streamlit frontend for the AI Customer Support Agent.
Talks to the FastAPI backend over HTTP.
"""
import os
import uuid
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Support Assistant", page_icon="💬", layout="wide")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "citations", "message_id"}]
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = set()

# ---------------- Sidebar: document upload ----------------
with st.sidebar:
    st.header("📄 Knowledge Base")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is not None and st.button("Upload & Index"):
        with st.spinner("Chunking, embedding, and indexing..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                resp = requests.post(f"{API_BASE_URL}/documents/upload", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']}")
            except Exception as e:
                st.error(f"Upload failed: {e}")

    st.divider()
    st.subheader("Indexed documents")
    try:
        docs_resp = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        docs_resp.raise_for_status()
        docs = docs_resp.json()["documents"]
        if docs:
            for d in docs:
                st.caption(f"• {d['filename']} — {d['chunks']} chunks")
        else:
            st.caption("No documents indexed yet.")
    except Exception:
        st.caption("Backend unreachable.")

    st.divider()
    if st.button("🔄 New conversation"):
        try:
            requests.post(
                f"{API_BASE_URL}/chat/reset",
                params={"thread_id": st.session_state.thread_id},
                timeout=10,
            )
        except Exception:
            pass
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.feedback_given = set()
        st.rerun()

# ---------------- Main chat area ----------------
st.title("💬 AI Customer Support Assistant")
st.caption("Ask a question about the uploaded documents. Answers include citations.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander(f"📚 Sources ({len(msg['citations'])})"):
                for c in msg["citations"]:
                    st.markdown(f"**[{c['id']}] {c['source']}** — page {c.get('page', '?')}")
                    st.markdown(f"> {c['snippet']}")
        if msg["role"] == "assistant" and msg.get("message_id"):
            mid = msg["message_id"]
            if mid not in st.session_state.feedback_given:
                col1, col2, _ = st.columns([1, 1, 8])
                if col1.button("👍", key=f"up_{mid}"):
                    requests.post(
                        f"{API_BASE_URL}/feedback",
                        json={
                            "message_id": mid,
                            "thread_id": st.session_state.thread_id,
                            "rating": "up",
                        },
                        timeout=10,
                    )
                    st.session_state.feedback_given.add(mid)
                    st.rerun()
                if col2.button("👎", key=f"down_{mid}"):
                    requests.post(
                        f"{API_BASE_URL}/feedback",
                        json={
                            "message_id": mid,
                            "thread_id": st.session_state.thread_id,
                            "rating": "down",
                        },
                        timeout=10,
                    )
                    st.session_state.feedback_given.add(mid)
                    st.rerun()
            else:
                st.caption("Thanks for the feedback!")

question = st.chat_input("Ask a question...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/chat",
                    json={"thread_id": st.session_state.thread_id, "question": question},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                st.markdown(data["answer"])
                if data["citations"]:
                    with st.expander(f"📚 Sources ({len(data['citations'])})"):
                        for c in data["citations"]:
                            st.markdown(f"**[{c['id']}] {c['source']}** — page {c.get('page', '?')}")
                            st.markdown(f"> {c['snippet']}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "citations": data["citations"],
                        "message_id": data["message_id"],
                    }
                )
            except Exception as e:
                st.error(f"Error: {e}")