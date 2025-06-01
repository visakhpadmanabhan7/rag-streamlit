def build_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context}

Question: {query}
Answer:"""
