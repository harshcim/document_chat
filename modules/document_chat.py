# library Importing
import os
import sys
import shutil
import streamlit as st  # type: ignore
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS 
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.doc_processing import load_data, split_data, save_embeddings
from utils.file_utils import list_uploaded_files, save_uploaded_file
from log.logger import setup_logger





# Load environment variables0
load_dotenv()   
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Model defining
rag_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.5, max_output_tokens=8192)




# Embedding loading
def load_embeddings_data(index_embedding):
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    db = FAISS.load_local(index_embedding, embeddings, allow_dangerous_deserialization=True)
    
    return db



# Defining function for chat functionality
def chat_with_doc(base_dir, uploaded_file):
        
    
    # Display currently uploaded files
    st.subheader("Uploaded Files")
    uploaded_files = list_uploaded_files()

    
    if uploaded_files:
        st.write("Currently available files in the directory:")
        for file in uploaded_files:
            st.write(f"- {file}")
    else:
        st.write("No files currently uploaded.")
        
        
        
    index_embedding = os.path.join(base_dir, "embeddings", "vector_store.faiss")

    index_path = os.path.join(base_dir,"embeddings", "vector_store.faiss", "index.faiss")
        
        
        
    if uploaded_file:
        save_uploaded_file(uploaded_file)
        st.success(f"File '{uploaded_file.name}' has been uploaded and stored.")
        
        # Trigger background processing if embeddings don't already exist
        if not os.path.exists(index_path):
            with st.spinner("Processing your files..."):
                
                documents = load_data()
                
                if documents:
                    st.write(f"Documents loaded: {len(documents)}")
                else:
                    st.warning("No documents were loaded.")
                
                if documents and any(doc.page_content.strip() for doc in documents):
                    text_chunks = split_data(documents)
                    # st.write(f"Documents loaded: {len(text_chunks)}")
                    save_embeddings(text_chunks)
                    st.success("You can now query the data.")
                else:
                    st.warning("No valid documents found to process.")
    else:
        st.write("Upload a file to create a knowledge base.")
        
        
        
        
    if os.path.exists(index_path):
        # If vector database exists, proceed with querying
        rag_prompt = """ 
        You are an advanced assistant designed to help users query and interpret complex industry-standard documents, technical guidelines, and compliance instructions. 
        Your role is to provide precise, contextually relevant, and actionable answers based solely on the document's content.

        Guidelines for Responding to Queries:

        1. **Understand User Intent**:
            - Accurately interpret user queries based on the provided context and question.
            - If the query is ambiguous, ask clarifying questions to refine the user's intent.
            - Determine if the query requires a short answer or a detailed response:
                - Short Answer: If the question is direct or seeks a concise explanation (e.g., 'What is section 3.1?'),
                provide a brief, specific answer.
                - Detailed Answer: If the query asks for an explanation, comparison, or breakdown 
                (e.g., 'Can you explain the steps for compliance in section 4?'), provide a comprehensive and structured response, possibly breaking it into manageable part
            - When the user provides detailed information about their product or scenario:
                - Identify the relevant aspects of the product (e.g., specifications, use case, compliance needs) to focus your response.
                - Match this information to the applicable guidelines or standards in the document.

        2. **Document-Specific Responses**:
            - Use the content of the provided documents to construct fact-based answers.
            - Reference section titles, numbers, or metadata for clarity (e.g., 'Section 5.3: Safety Requirements').
            - If the requested information isn’t explicitly found in the document, clearly state, 'This information is not available in the provided documents.'

        3. **Output Formatting and Style**:
            - Structure responses logically and professionally. Use bullet points, numbered steps, or tables when answering procedural or comparison-based questions.
            - Maintain industry-appropriate terminology and tone while ensuring accessibility for users with varying expertise levels.
            - For complex queries, break down the response into manageable parts, summarizing key points before diving into details.

        4. **Handle Contextual Complexity**:
            - Address multi-step or compound queries comprehensively.
            - Summarize related sections or cross-reference similar standards if it enhances user understanding.

        5. **Metadata and Enhanced Retrieval**:
            - Leverage document metadata, such as section titles, tags, or keywords, to refine and enhance the relevance of your response.
            - Indicate the source section or subsection for every provided answer to improve traceability.

        6. **Clarify Limitations**:
            - If a query requires interpretation or decisions beyond the document’s content, clearly state that your responses are fact-based and do not provide subjective opinions.
            - Avoid making assumptions; if a query requires external knowledge, state this explicitly.

        Response Format Guidelines:
            - **Bullet Points**: Use bullet points for clarity when listing multiple details or points.
            - **Direct Answers**: Provide direct answers for questions about specific clauses, sections, or standards.
            - **Keep It Concise**: Stick to brief and relevant responses, maintaining focus on the user’s question.
            - **For Detailed Responses**: Provide thorough explanations, breaking down steps or sections where needed.

        Context:  
        {context}

        Question:  
        {question}
        
        """
        
        prompt = PromptTemplate(template=rag_prompt, input_variables=["context", "question"])

        db = load_embeddings_data(index_embedding)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=rag_model,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        # Query input box
        # question = st.text_input("Ask a question related to the documents:")
        question = st.text_area("Ask a question related to the documents:", height=150)

        if question:
            with st.spinner("Fetching response..."):
                response = qa_chain.invoke({"query": question})
                st.write("Response:")
                st.write(response['result'])
                
    else:
        # If vector database does not exist, prompt the user to upload content
        st.warning("No Knowledgebase found. Please upload document.")
        
        
        
        
        
        
    
    
    
        
        
    
    