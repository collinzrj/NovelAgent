## Novel Agent
It's difficult for current language model to process novels with millions of words. Most llm api has a context window size less than 1 million tokens, and they don't even support RAG of too many words. Moreover, naive RAG systems don't perform well on answering questions on novel cause they don't extract the information needed to answer the questions.

This agent improves RAG in an iterative way, it refines the question to multiple queries, using these queries to extract relevant information, and then refine the queries based on the context (extracted information). Finally, the agent used these refined queries to extract relevant information to answer the questions.

On experiment on the novel "诡秘之主"

On the question "有哪几个序列会改变性别"

RAG answers "改变性别的序列有很多种方式和情况，具体取决于你所指的上下文或领域。如果你是在谈论生物学上的性别转变，那么通常涉及到一些动物，如某些鱼类、两栖类、爬行类，它们可以在一生中自然改变性别。这种现象被称为性别转变或性别多态性。常见的例子包括雌雄同体的动物和能进行性别转变的鱼类，如横带双锯鱼和珊瑚礁石斑鱼等。\n\n在其他领域，例如文学、神话或科幻小说中，改变性别可能是通过魔法、超自然力量或科技手段实现的。这些情景通常是虚构的，涵盖了从神话故事中的神变性别到科幻小说中通过技术手段改变性别的情景。\n\n如果你指的是具体的文本或领域，请提供更多信息，以便我能给出更准确的回答。"

Agent answers "根据收集到的信息，有关序列改变性别的提及位于以下段落：\n\n“大帝，你不会变女性，但你上过一个魔女，也许不止一个……‘果然有女变男的途径，而且在我预测的那几个里面……‘猎人’代表着战争，会在序列4导致女士变性……这有点不对啊，魔女途径是在序列7‘女巫’这个阶段啊，竟然没有严格对应……”克莱恩越想越体会到一种扭曲和疯狂感，这是由极致的不对称带来的。\n\n从这段描述中，我们可以得知至少有两个序列途径会涉及到性别的改变：一个是序列4“猎人”，它会导致女性变性；另一个则是序列7“女巫”，虽然这里并未直接提到性别变化，但从上下文推测，这似乎是另一个与性别改变相关的序列。因此，基于提供的内容，至少有“猎人”和“女巫”这两个序列途径会涉及到性别的改变。"

RAG is unable to extract relevant information to answer the question, while agent answers the question correctly

## Getting Started
By default the code is set to answer questions on 诡秘之主

set environment variable OPENAI_API_KEY, and install requirements
```
export OPENAI_API_KEY=your_api_key
pip install -r requirements.txt
```
run
```
python try_langchain.py
```
It will ask you to enter a question, then answer it


you can also compare it with rag by
```
python try_rag.py
```

## TODO
[] improve the prompts
[] make the agent act adaptively