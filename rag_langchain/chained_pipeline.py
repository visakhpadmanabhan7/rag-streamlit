import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import tempfile

# 🔧 Build retriever from uploaded PDF
@st.cache_resource
def load_retriever_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)

    return vectordb.as_retriever()

# 💬 LLaMA via Ollama
llm = ChatOllama(model="llama3", temperature=0.4)

# ⛓ Prompt 1 – Summary
summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text:\n\n{text}\n\nSummary:"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

# ⛓ Prompt 2 – Tone Analysis
analysis_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Analyze the tone and writing style of this summary:\n\n{summary}\n\nAnalysis:"
)
analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt, output_key="analysis")

# ⛓ Prompt 3 – Rewriting
rewrite_prompt = PromptTemplate(
    input_variables=["summary", "analysis"],
    template="""
Rewrite the summary below for a 10-year-old child, using a simpler tone.

Summary:
{summary}

Tone Analysis:
{analysis}

Rewritten:
"""
)
rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt, output_key="rewritten")

# 🔗 Full Pipeline
pipeline = SequentialChain(
    chains=[summary_chain, analysis_chain, rewrite_chain],
    input_variables=["text"],
    output_variables=["summary", "analysis", "rewritten"],
    verbose=False
)

# 🌐 Streamlit UI
st.set_page_config(page_title="RAG + LLaMA Prompt Chain", page_icon="🧠")
st.title("📄 Ask Questions from Your PDF + Rewrite for Kids")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    retriever = load_retriever_from_pdf(uploaded_file)
    user_query = st.text_input("Ask a question from your PDF:")

    if user_query:
        with st.spinner("Retrieving and processing..."):
            docs = retriever.get_relevant_documents(user_query)
            context = "\n\n".join(doc.page_content for doc in docs)

            result = pipeline({"text": context})

        st.markdown("### 📄 Summary")
        st.write(result["summary"])

        st.markdown("### 🧠 Tone Analysis")
        st.write(result["analysis"])

        st.markdown("### 👶 Rewritten for Kids")
        st.write(result["rewritten"])