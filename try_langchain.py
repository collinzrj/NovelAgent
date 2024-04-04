from operator import itemgetter
from langchain import hub
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import HumanMessage


vectorstore = FAISS.load_local("novel_embedding", OpenAIEmbeddings())

retriever = vectorstore.as_retriever()

tool = create_retriever_tool(
    retriever,
    "novelreader",
    "You are going to read a novel and answer questions.",
)
tools = [tool]


llm = ChatOpenAI(model_name='gpt-4-turbo-preview')

def first_round_retrieve(question):
    template = """You are an ai agent with access to a novel for which you can perform search on by retrieval augmented generation, 
    and the user will ask you questions on the novels. When you received the question, you will think step by step what information 
    you need to answer the question, and then generate queries to query the novel. Generate at least 5 queries. You will answer in Chinese.
    问题: {question}
    Start with "1:"
    """
    prompt = ChatPromptTemplate.from_template(template)
    prompt = prompt.format_messages(question=question)


    llm_result = llm.invoke(prompt)
    print("First round proposed queries:")
    print(llm_result.content, '\n')

    pairs = ''
    for query in llm_result.content.split('\n'):
        # pairs += f'{query}\n'
        relevant_docs = retriever.get_relevant_documents(query)
        for doc in relevant_docs:
            pairs += f'{doc.page_content}\n'
    return pairs

def refine_question(pairs, question):
    template = """You are an ai agent with access to a novel for which you can perform search on by retrieval augmented generation, 
    and the user will ask you questions on the novels. When you received the question, you will think step by step what information 
    you need to answer the question, and then generate queries to query the novel. Now you have gathered some queries and their relevant docs:
    {doc}
    Now you will rewrite the user's question, to make it include more details and contain more information, the question should be longer than 50 words, you will answer in Chinese
    问题: {question}
    New Question:
    """
    prompt = ChatPromptTemplate.from_template(template)
    prompt = prompt.format_messages(doc=pairs, question=question)

    llm_result = llm.invoke(prompt)
    print("Refined question:\n" + llm_result.content, '\n')
    return llm_result

def second_round_retrieve(pairs, question):
    template = """You are an ai agent with access to a novel for which you can perform search on by retrieval augmented generation, 
    and the user will ask you to write a report on some questions about content of the novels. The user will give you a report outline, 
    When you received the report outline, you will think step by step what information 
    you need to answer the question, and then generate queries to query the novel. Now you have gathered some queries and their relevant docs:
    {doc}
    Now you will refine you queries with the doc you have gathered to include all info you need to answer the question. Generate at least 5 queries.
    You will answer in Chinese.
    问题: {question}
    Start with "1:"
    """
    prompt = ChatPromptTemplate.from_template(template)
    prompt = prompt.format_messages(doc=pairs, question=question)


    llm_result = llm.invoke(prompt)
    print('Refined queries:\n' + llm_result.content, '\n')

    pairs = ''
    for query in llm_result.content.split('\n'):
        # pairs += f'{query}\n'
        relevant_docs = retriever.get_relevant_documents(query)
        for doc in relevant_docs:
            pairs += f'{doc.page_content}\n'
    return pairs

def answer_question_with_doc(pairs, question):
    # print(pairs)
    template2 = """You are an ai agent with access to a novel for which you can perform search on by retrieval augmented generation, 
    and the user will ask you questions on the novels.
    Now you have gathered severals queries and documents relevant to them listed below:
    {doc}
    Analyze the content you have gathered step by step to answer this question, make sure you effectively use the information provided in the content.
    Question:
    {question}
    """
    prompt = ChatPromptTemplate.from_template(template2)
    prompt = prompt.format_messages(doc=pairs, question=question)
    llm_result = llm.invoke(prompt)
    print('Final answer:\n' + llm_result.content, '\n')


def make_plan(pairs, question):
    # print(pairs)
    template2 = """You are an ai agent with access to a novel for which you can perform search on by retrieval augmented generation, 
    and the user will ask you questions on the novels.
    Now you have gathered severals queries and documents relevant to them listed below:
    {doc}
    Given this context, construct a step by step plan to answer the question. You will answer in Chinese. 
    Question:
    {question}
    Plan:
    """
    prompt = ChatPromptTemplate.from_template(template2)
    prompt = prompt.format_messages(doc=pairs, question=question)
    llm_result = llm.invoke(prompt)
    print('Plan:\n' + llm_result.content, '\n')
    return llm_result



if __name__ == "__main__":
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
    print("")
    pairs = first_round_retrieve(question)
    new_question = refine_question(pairs, question)
    pairs = second_round_retrieve(pairs, new_question)
    answer_question_with_doc(pairs, new_question)