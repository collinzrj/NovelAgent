import regex
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

with open('guimi.txt', encoding='utf-16') as f:
    text = f.read()

# Find all occurrences of "第x章" where x is a digit
pattern = r"第\p{script=Han}+章"
matches = regex.finditer(pattern, text)

# Get the indices
indices = [(match.start(), match.end()) for match in matches]

num_chap = len(indices)

chapters = []
for idx in range(num_chap):
    if idx == num_chap - 1:
        chapter = text[indices[idx][0]:]
    else:
        chapter = text[indices[idx][0]:indices[idx+1][0]]
    chapters.append(chapter)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.create_documents(chapters)


vectorstore = FAISS.from_documents(
    texts, embedding=OpenAIEmbeddings()
)
vectorstore.save_local("guimi_embedding")