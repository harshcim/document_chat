# Library Importing
import os
import sys
import shutil
import streamlit as st  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from modules.document_chat import chat_with_doc
from utils.file_utils import delete_all_files_and_directories
from utils.doc_processing import load_data, split_data, save_embeddings
from log.logger import setup_logger


# Defining base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# logger setup
logger = setup_logger("main_script")




# Streamlit interface
st.title("Chat with Document")
st.write("Upload PDF or DOCX files to build your dynamic knowledge base")



# File uploader for PDF and ZIP files
uploaded_file = st.file_uploader(
    "Upload a PDF, DOCX",
    type=["pdf", "docx"],  # Updated to accept 'zip' files
    accept_multiple_files=False,
)


chat_with_doc(base_dir, uploaded_file)


    
if st.button("Clear All Uploaded Data"):
    delete_all_files_and_directories()
    st.success("All uploaded files have been cleared!")
