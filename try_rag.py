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


vectorstore = FAISS.load_local("novel_embedding", OpenAIEmbeddings())

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
print("""Example Questions:
- 克莱恩和格尔曼斯帕罗是什么关系
- 克莱恩和心理炼金会的矛盾是怎么产生的
- 原初魔女为什么仇视女性魔女
- 有哪几个序列会改变性别
- 克莱恩和天尊是什么关系
- 克莱恩的穿越是怎么一回事
- 罗塞塔大帝遭遇了什么困难
- 小说中出现了哪些穿越者
- 远古太阳神和真实造物主是什么关系
- 小说设定中的历史是什么样的，请简述小说中历史发展过程
- 班西港发生了什么事情
- 玫瑰学派和愚者教会有什么关系
- 请总结小说中所有隐秘组织""")
question = input("Please enter your question:")

result = agent_executor.invoke({
    "input": question
})


print("Final answer:")
print(result)