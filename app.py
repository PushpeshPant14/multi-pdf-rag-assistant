import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Load .env from script directory or root
load_dotenv(Path(__file__).parent / ".env")
load_dotenv()

st.set_page_config(page_title="Multi-PDF RAG Chatbot", page_icon="📄", layout="wide")

PROMPT_TEMPLATE = """
You are a helpful document assistant. Answer the user's question using only the retrieved context.
If the answer is not present in the context, say that the document does not contain enough information.
Always provide a concise answer followed by a short 'Sources' section listing the page numbers and chunk references you used.

Context:
{context}

Question:
{question}

Answer:
"""


def extract_text_from_pdfs(uploaded_files) -> List[Document]:
    documents = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": uploaded_file.name,
                            "page": page_num,
                        },
                    )
                )
    return documents


def chunk_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = idx
    return chunks


def build_vector_store(chunks: List[Document], embeddings, store_type: str, persist_dir: str):
    if store_type == "FAISS":
        return FAISS.from_documents(chunks, embeddings)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)


def format_sources(source_docs: List[Document]) -> str:
    seen = []
    for doc in source_docs:
        ref = f"{doc.metadata.get('source', 'Unknown')} - page {doc.metadata.get('page', '?')} - chunk {doc.metadata.get('chunk_id', '?')}"
        if ref not in seen:
            seen.append(ref)
    return "\n".join(f"- {item}" for item in seen)


def build_qa_chain(vector_store, model_name: str = "gemini-2.5-flash"):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def answer_question(query: str) -> Tuple[str, List[Document]]:
    qa_chain = st.session_state.get("qa_chain")
    if qa_chain is None:
        return "Please process at least one PDF first.", []
    result = qa_chain.invoke({"query": query})
    return result["result"], result.get("source_documents", [])


def sidebar():
    st.sidebar.title("⚙️ Configuration")
    api_key = st.sidebar.text_input("Google Gemini API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    model_name = st.sidebar.selectbox("Gemini Model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest"], index=0)
    store_type = st.sidebar.selectbox("Vector Store", ["FAISS", "ChromaDB"])
    chunk_size = st.sidebar.slider("Chunk Size", 300, 2000, 900, 100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", 0, 400, 150, 10)
    return store_type, chunk_size, chunk_overlap, model_name


def main():
    st.title("📄 Multi-PDF RAG Chatbot")
    st.caption("Upload multiple PDFs, build embeddings with FAISS or ChromaDB, and chat with your documents using Gemini.")

    store_type, chunk_size, chunk_overlap, model_name = sidebar()

    with st.expander("Project features", expanded=True):
        st.markdown(
            """
            - PDF upload and parsing with page-level metadata
            - Recursive chunking pipeline for semantic retrieval
            - Choice between **FAISS** and **ChromaDB** vector stores
            - Gemini-powered answers constrained to retrieved context
            - Source-aware output with file name, page number, and chunk reference
            """
        )

    uploaded_files = st.file_uploader("Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        process_clicked = st.button("Process Documents", use_container_width=True)
    with col2:
        clear_clicked = st.button("Reset Session", use_container_width=True)

    if clear_clicked:
        for key in ["qa_chain", "vector_store", "chunks"]:
            st.session_state.pop(key, None)
        st.success("Session cleared.")

    if process_clicked:
        if not os.getenv("GOOGLE_API_KEY"):
            st.error("Please provide your Google Gemini API key.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            with st.spinner("Reading PDFs and building vector index..."):
                raw_docs = extract_text_from_pdfs(uploaded_files)
                chunks = chunk_documents(raw_docs, chunk_size, chunk_overlap)
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                )
                persist_dir = tempfile.mkdtemp(prefix="chromadb_")
                vector_store = build_vector_store(chunks, embeddings, store_type, persist_dir)
                qa_chain = build_qa_chain(vector_store, model_name=model_name)
                st.session_state["vector_store"] = vector_store
                st.session_state["qa_chain"] = qa_chain
                st.session_state["chunks"] = chunks
            st.success(f"Processed {len(raw_docs)} pages into {len(chunks)} chunks using {store_type}.")

    if "chunks" in st.session_state:
        with st.expander("Chunk preview"):
            preview_count = min(5, len(st.session_state["chunks"]))
            for doc in st.session_state["chunks"][:preview_count]:
                st.markdown(
                    f"**{doc.metadata.get('source')} | Page {doc.metadata.get('page')} | Chunk {doc.metadata.get('chunk_id')}**\n\n{doc.page_content[:350]}..."
                )

    query = st.text_input("Ask a question about the uploaded document(s)")
    if st.button("Get Answer", use_container_width=True):
        response, source_docs = answer_question(query)
        st.subheader("Answer")
        st.write(response)
        st.subheader("Retrieved Sources")
        if source_docs:
            st.code(format_sources(source_docs), language="text")
        else:
            st.info("No sources available.")

    st.markdown("---")
    st.markdown(
        "**Example questions:** Summarize the main topic. | What are the key findings? | Which page mentions the conclusion?"
    )


if __name__ == "__main__":
    main()
