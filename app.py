import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="wide"
)

CHROMA_DIR = "chromaDB"


# ---------- Cached resources ----------

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
            ),
            (
                "human",
                """Context:
{context}

Question:
{question}
"""
            )
        ]
    )


# ---------- Core logic ----------

def build_vectorstore_from_pdf(uploaded_file):
    """Save uploaded PDF to a temp path, chunk it, and build a fresh Chroma store."""
    # Wipe any previous collection so old book content doesn't leak into new chats
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(docs)

        embeddings = get_embeddings()

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )
        return vectorstore, len(docs), len(chunks)
    finally:
        os.remove(tmp_path)


def get_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )


def answer_question(retriever, llm, prompt, question):
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(final_prompt)
    return response.content, docs


# ---------- Session state ----------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "book_name" not in st.session_state:
    st.session_state.book_name = None


# ---------- Sidebar: upload ----------

with st.sidebar:
    st.header("📄 Upload your book")
    uploaded_file = st.file_uploader(
        "Upload a PDF to chat with",
        type=["pdf"]
    )

    process_clicked = st.button(
        "Process document",
        type="primary",
        disabled=uploaded_file is None
    )

    if process_clicked and uploaded_file is not None:
        with st.spinner("Reading and indexing your document... this can take a moment"):
            vectorstore, num_pages, num_chunks = build_vectorstore_from_pdf(uploaded_file)
            st.session_state.vectorstore = vectorstore
            st.session_state.retriever = get_retriever(vectorstore)
            st.session_state.book_name = uploaded_file.name
            st.session_state.messages = []  # fresh chat for the new document

        st.success(f"Indexed {num_pages} pages into {num_chunks} chunks ✅")

    st.divider()

    if st.session_state.book_name:
        st.markdown(f"**Current document:** {st.session_state.book_name}")
    else:
        st.markdown("*No document loaded yet.*")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


# ---------- Main area: chat ----------

st.title("Chat with your PDF 📄💬")

if st.session_state.retriever is None:
    st.info("Upload a PDF from the sidebar and click **Process document** to get started.")
else:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask something about your document...")

    if user_query:
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm = get_llm()
                prompt = get_prompt()
                answer, source_docs = answer_question(
                    st.session_state.retriever, llm, prompt, user_query
                )
                st.markdown(answer)

                with st.expander("View retrieved context"):
                    for i, doc in enumerate(source_docs, start=1):
                        page = doc.metadata.get("page", "N/A")
                        st.markdown(f"**Chunk {i} (page {page}):**")
                        st.text(doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""))

        st.session_state.messages.append({"role": "assistant", "content": answer})