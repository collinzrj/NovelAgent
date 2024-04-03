from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
from langchain_core.messages import HumanMessage


# vectorstore = FAISS.from_texts(
#     ["harrison works in Yixing"], embedding=OpenAIEmbeddings()
# )
# vectorstore.save_local("faiss_index")
vectorstore = FAISS.load_local("guimi_embedding", OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

tool = create_retriever_tool(
    retriever,
    "novelreader",
    "You are going to read a novel and answer questions.",
)
tools = [tool]

prompt = hub.pull("hwchase17/openai-tools-agent")

llm = ChatOpenAI(model_name='gpt-4-turbo-preview')
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

result = agent_executor.invoke({
    "input": "小说设定中的历史是什么样的，请简述小说中历史发展过程"
})


print(result)