from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from pdf_tool import create_pdf_qa

# Load PDF tool
pdf_path = "path"
pdf_qa = create_pdf_qa(pdf_path)

# Define tools
tools = [
    Tool(name="PDF_QA", func=pdf_qa.run, description="Useful for answering questions about the PDF"),
    Tool(name="Calculator", func=lambda x: str(eval(x)), description="Useful for math questions")
]

# Set up LLM + memory
llm = ChatOllama(model="llama3", temperature=0.4)
memory = ConversationBufferMemory(memory_key="chat_history")

# Agent setup
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# CLI loop
print("🧠 PDF Agent is ready. Ask anything or type 'exit'.")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "exit":
        break
    result = agent.run(user_input)
    print("🤖 Agent:", result)