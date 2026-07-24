from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st


load_dotenv()

loader = PyPDFLoader(".\The Ultimate Python Handbook.pdf")
docs = loader.load()

# print(len(documents))

splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitters.split_documents(docs)

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
  
query = input("Enter your question: ")

documents = vectorstore.similarity_search(query=query, k=4)

# print(len(documents[1].page_content)) 

context = ""
for doc in documents:
    context = context + doc.page_content + "\n\n"


        
#prompt = PromptTemplate(
#    template="""
#        You are a helpful assistant.
#        Answer ONLY from the provided transcript context.
#        If the context is insufficient, just say you don't know.
#
#        Context:{context}
#
#        Question:{query}
#        """,
#        input_variables=["context", "query"]
#    )

prompt = f""""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context:{context},


        Question:{query}"""


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6)

# finalprompt = prompt.invoke({"context": context, "query": query})
response = llm.invoke(prompt)

print(response.content)