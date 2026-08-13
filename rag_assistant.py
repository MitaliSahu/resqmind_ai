import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document

load_dotenv()

FAISS_INDEX_PATH = "models/faiss_index"

def setup_rag():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_actual_api_key_here":
        return None

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Initialize emergency guideline data if local vector store is empty
    if not os.path.exists(FAISS_INDEX_PATH) or not os.listdir(FAISS_INDEX_PATH):
        docs = [
            Document(page_content="Critical Priority: SpO2 < 88%, severe chest pain, loss of consciousness, severe trauma. Requires immediate ICU or resuscitation room."),
            Document(page_content="High Priority: Heart Rate > 120 bpm, high fever (> 102 F), severe abdominal pain. Requires doctor evaluation within 15 minutes."),
            Document(page_content="Medium/Low Priority: Normal vitals, mild fever, minor cuts, cold symptoms. Assigned to standard waiting queue.")
        ]
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
    else:
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa_chain

def ask_rag_assistant(query):
    try:
        qa_chain = setup_rag()
        if qa_chain is None:
            return "⚠️ Please add a valid GOOGLE_API_KEY to your .env file to activate the RAG assistant."
        response = qa_chain.invoke(query)
        return response['result']
    except Exception as e:
        return f"Error connecting to AI Assistant: {str(e)}"