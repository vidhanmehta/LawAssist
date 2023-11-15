import streamlit as st
import os
from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from st_files_connection import FilesConnection

st.title("Law Assist")
st.write("Ask me legal questions, and I'll provide answers!")
st.write("Created by: Vidhan Mehta, Sumith Sigtia, Shabiul Hasnain Siddiqui, Swathi")

# Define the path to your PDF file
pdf_path = "merge.pdf"

# Load PDF and split by pages
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

# Convert pages to chunks
chunks = pages

# Create embeddings model
os.environ["OPENAI_API_KEY"] = "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"
embeddings = OpenAIEmbeddings()

# Create vector database
db = FAISS.from_documents(chunks, embeddings)

# Load QA chain
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")

# Create conversation chain
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

# Initialize chat history
chat_history = []

def generate_response(user_input):
    result = qa({"question": user_input, "chat_history": chat_history})
    chat_history.append(("You:", user_input))
    chat_history.append(("Bot:", result['answer']))
    return result['answer']

user_input = st.text_input("You:", value="")

if user_input:
    if user_input.lower() == 'exit':
        st.write("Thank you for using the LAW Assist chatbot!")
    else:
        bot_response = generate_response(user_input)
        
        for speaker, text in chat_history:
            st.text(speaker)
            st.write(text)
        
        # Clear the input field
        user_input = ""
