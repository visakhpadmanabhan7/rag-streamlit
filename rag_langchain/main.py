import streamlit as st
from load_and_embed import load_and_embed
from qa_chain import create_qa_chain
import tempfile
import os

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("📄 Chat with your PDF")
st.write("Ask questions about your uploaded PDF.")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    # Embed and create chain
    with st.spinner("🔄 Processing PDF and building vector index..."):
        vectordb = load_and_embed(pdf_path)
        qa = create_qa_chain(vectordb)

    st.success("✅ PDF processed! Ask your questions below.")

    # Input box for user query
    user_question = st.text_input("Ask a question:")

    if user_question:
        with st.spinner("💬 Thinking..."):
            result = qa(user_question)

        st.markdown("### 🧠 Answer")
        st.write(result['result'])

        st.markdown("### 🔍 Sources")
        for doc in result['source_documents']:
            source = doc.metadata.get("source", "Unknown")
            st.write("-", source)