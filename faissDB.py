import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.embeddings import HuggingFaceEmbeddings
load_dotenv() ## aloading all the environment variable

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")



### from langchain_cohere import CohereEmbeddings

# Set embeddings
embd = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# Docs to index
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load documents from URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

# Create vector store
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=embd,
)
retriever = vectorstore.as_retriever()

print(f"Loaded {len(docs_list)} documents from {len(urls)} URLs")
print(f"Split into {len(doc_splits)} chunks")
print("Vector store created successfully!")

# Save vector store to directory
db_dir = "faiss_index"
vectorstore.save_local(db_dir)
print(f"Vector store saved to '{db_dir}' directory")
