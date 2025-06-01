def search(query, index, chunks, embed_model, k=10):
    query_vec = embed_model.encode([query])
    distances, indices = index.search(query_vec, k)
    return [chunks[i] for i in indices[0]]
