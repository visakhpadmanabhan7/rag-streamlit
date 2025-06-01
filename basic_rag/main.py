from pdf_loader import load_pdf, chunk_text
from embed_store import embed_chunks, build_faiss_index, model
from retriever import search
from prompt_builder import build_prompt
from llama_client import ask_llama

import os

# ----------- Step 1: Load and chunk document -----------
PDF_PATH = "basic_rag/your_file.pdf"

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"File not found: {PDF_PATH}")

print(f"\n📄 Loading PDF: {PDF_PATH}")
text = load_pdf(PDF_PATH)
chunks = chunk_text(text)

# ----------- Step 2: Embed and build index -----------
print("🔄 Embedding text and building FAISS index...")
embeddings = embed_chunks(chunks)
index = build_faiss_index(embeddings)

# ----------- Diagnostics -----------
print("\n📊 Index Stats:")
print("• Number of vectors:", index.ntotal)
print("• Vector dimension:", index.d)
memory_mb = index.ntotal * index.d * 4 / (1024 ** 2)
print(f"• Estimated index memory: {memory_mb:.2f} MB")

# ----------- Step 3: Multi-turn Chat Loop -----------
chat_history = []

print("\n🤖 Ask questions based on your document (type 'exit' to quit)")
while True:
    query = input("\n🧑 You: ")
    if query.lower() == "exit":
        print("👋 Exiting chat.")
        break

    top_chunks = search(query, index, chunks, model)

    # Build RAG context prompt
    context_prompt = build_prompt(query, top_chunks)

    # Append chat history to prompt (basic memory)
    if chat_history:
        history_text = "\n".join([f"User: {q}\nAI: {a}" for q, a in chat_history])
        full_prompt = f"{history_text}\n\n{context_prompt}"
    else:
        full_prompt = context_prompt

    response = ask_llama(full_prompt)
    print("🤖 AI:", response.strip())

    # Save to memory
    chat_history.append((query, response.strip()))
