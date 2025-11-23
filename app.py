# Desactiva la verificación SSL para descargar el modelo

import streamlit as st
import tempfile
import os

os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
# LangChain and Ollama libraries for AI and PDF processing
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings



# --- 1. UI CONFIGURATION ---
st.set_page_config(
    page_title="Local RAG - Secure Doc Chat",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Local Secure Document Assistant (RAG)")
st.markdown("""
**Chat with your confidential documents securely.**
This tool processes your PDFs locally using Llama 3. Your data never leaves this machine.
""")

# --- 2. SIDEBAR (INPUTS) ---
with st.sidebar:
    st.header("📁 Control Panel")
    uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

    st.divider()
    model_name = st.selectbox(
        "AI Engine (Ollama)",
        ["llama3.1", "mistral", "gemma3n"],
        index=0,
        help="Select the model installed on your local machine."
    )

    st.info("ℹ️ Ensure Ollama is running in the background.")

# --- 3. MAIN LOGIC ---
if uploaded_file is not None:

    # Save file temporarily so PyPDFLoader can access it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    try:
        with st.spinner('🧠 Analyzing document and generating vector memory...'):

            # A. Load and Chunking
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Split text into 1000-character chunks with overlap
            # This is vital to maintain context between pages.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)

            # B. Indexing (Embeddings + Vector Store)
            # Convert text to numerical vectors locally
            embeddings = OllamaEmbeddings(model=model_name)
            #embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()

            # C. Brain Configuration (LLM)
            llm = ChatOllama(model=model_name, temperature=0.1)  # Low temp = high precision

            # System Prompt: Strict instructions for the model
            system_prompt = (
                "You are an expert analyst. Use ONLY the following context to answer "
                "the user's question. If the answer is not in the context, state that you do not know. "
                "Keep the answer professional and concise."
                "\n\n"
                "Document Context:\n{context}"
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])

            # Create the reasoning chain
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            st.toast(f"Document indexed successfully: {len(splits)} chunks.", icon="✅")

        # --- 4. CHAT INTERFACE ---
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display session history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Capture user input
        if user_question := st.chat_input("Ex: What are the termination clauses?"):
            # 1. Display user question
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            # 2. Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Consulting the document..."):
                    response = rag_chain.invoke({"input": user_question})
                    st.markdown(response["answer"])

            # 3. Save response to history
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

    finally:
        # Cleanup: remove temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    # Initial State (No file)
    st.markdown("### ⬅️ Step 1: Upload a PDF in the sidebar")
    st.caption("The system will process the text and allow you to ask questions about its content.")


