import streamlit as st
from pdf_loader import chunk_text
from embed_store import embed_chunks, build_faiss_index, model
from retriever import search
from prompt_builder import build_prompt
from llama_client import ask_llama

from PyPDF2 import PdfReader

st.set_page_config(page_title="📄 Chat with Your PDF", layout="wide")
st.title("📄 Chat with Your PDF (RAG + LLaMA 3)")
st.caption("Upload a PDF, ask questions, and get local AI answers!")

# --------- Upload PDF ----------
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

def load_pdf_from_upload(file):
    reader = PdfReader(file)
    return "".join(page.extract_text() for page in reader.pages if page.extract_text())

if uploaded_file:
    # Process and cache chunks/index only once
    if "chunks" not in st.session_state:
        text = load_pdf_from_upload(uploaded_file)
        st.session_state.chunks = chunk_text(text)

    if "index" not in st.session_state:
        embeddings = embed_chunks(st.session_state.chunks)
        st.session_state.index = build_faiss_index(embeddings)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Input box
    query = st.text_input("Ask something from your PDF:")

    if query:
        top_chunks = search(query, st.session_state.index, st.session_state.chunks, model)
        context_prompt = build_prompt(query, top_chunks)

        # Append memory to prompt
        if st.session_state.chat_history:
            history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in st.session_state.chat_history])
            full_prompt = f"{history_text}\n\n{context_prompt}"
        else:
            full_prompt = context_prompt

        response = ask_llama(full_prompt).strip()
        st.session_state.chat_history.append((query, response))

    # Display chat
    for q, a in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {q}")
        st.markdown(f"**🤖 AI:** {a}")

else:
    st.info("👆 Upload a PDF file to begin chatting.")
