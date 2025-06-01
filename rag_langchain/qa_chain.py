from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def create_qa_chain(vectordb):
    llm = Ollama(model="llama3")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    return qa_chain
