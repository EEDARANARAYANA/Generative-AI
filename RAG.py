import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
groq_api_key = os.getenv("groq_api_key")

def save_uploaded_file(uploaded_file):
    temp_file_path=f"./temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return temp_file_path

#Streamlit UI
st.title("AI-Powered Document Q&A(RAG)")
st.write("Upload a PDF and ask questions about its content")

#File Uploader
uploaded_file = st.file_uploader("Upload File", type=["pdf","docx"])

if uploaded_file is not None:
    
    file_path=save_uploaded_file(uploaded_file)

    if file_path.endswith(".pdf"):
        loader=PyPDFLoader(file_path)
    
    elif file_path.endswith(".docx"):
        loader=Docx2txtLoader(file_path)

    documents=loader.load()

    #Split text into smaller chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs=text_splitter.split_documents(documents)

    #Convert text into embeddings and store in chromaDB
    embeddings = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
    vector_db=FAISS.from_documents(docs, embeddings)
    retriever=vector_db.as_retriever()

    #Load LLM and create RAG pipeline
    model = ChatGroq(model="gemma2-9b-it", groq_api_key=groq_api_key)
    qa_chain=RetrievalQA.from_chain_type(llm=model, retriever=retriever)

    #User query input
    query=st.text_input("Ask a question about the document:")
    if query:
        response=qa_chain.invoke(query)
        st.write("AI response")
        st.write(response)