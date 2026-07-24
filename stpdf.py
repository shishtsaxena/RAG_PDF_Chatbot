from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import streamlit as st
from time import sleep

load_dotenv()


llm = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.6)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

def documentprocess(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    
    splitters = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitters.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
        
    st.session_state.vectorstore = vectorstore
    st.session_state.pdf_docs = True
            

st.subheader("PDF Chatbot")

if "pdf_docs" not in st.session_state:
    st.session_state.pdf_docs = False

if not st.session_state.pdf_docs:
    file = st.file_uploader(label= "please pdf file to upload kriyea" , type="pdf")
    if file :
        with open("uploaded_file.pdf", "wb") as f:
            f.write(file.getvalue())

        with st.spinner("Processing..."):
            documentprocess("./uploaded_file.pdf")    

        st.markdown("PDF Uploaded Successfully")
        sleep(2)
        st.rerun()    


if st.session_state.pdf_docs and st.session_state.vectorstore:

    for onemessage in st.session_state.messages:
        role = onemessage["role"]
        content = onemessage["content"]
        st.chat_message(role).markdown(content)


    query = st.chat_input("ask anything about your PDF: ")
    if query:

        st.session_state.messages.append({"role": "user", "content": query})    
        
        st.chat_message("user").markdown(query)

#        greetings = ["hi", "hello", "hey", "good morning", "good evening"]
 #       if query.lower().strip() in greetings:
  #              st.chat_message("assistant").markdown(
   #             "Hello! 👋 I'm your PDF Chatbot. Ask me anything about your uploaded PDF."
    #            )

        #else:
           # documents = st.session_state.vectorstore.similarity_search(query=query, k=4)


        documents = st.session_state.vectorstore.similarity_search(query=query, k=4)

        context = ""
        for doc in documents:
            context = context + doc.page_content + "\n\n"
        
        prompt = f""""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, just say you don't know.

        Context:{context}

        Question:{query}"""
        
        response = llm.invoke(prompt)
        
          
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        st.chat_message("assistant").markdown(response.content)
        