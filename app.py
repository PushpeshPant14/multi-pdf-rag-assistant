# SQLite compatibility patch for ChromaDB on Streamlit Cloud (Linux)
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except (ImportError, KeyError):
    pass

import os
import tempfile
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Generator

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS, Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Environment & Page Setup
load_dotenv(Path(__file__).parent / ".env")
load_dotenv()

st.set_page_config(
    page_title="Multi-PDF RAG Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Premium Custom CSS Styling
CUSTOM_CSS = """
<style>
/* Modern Fonts & Base Styling */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Gradient Header */
.main-header {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.3rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

/* Glassmorphism Metric Cards */
.metric-container {
    display: flex;
    gap: 12px;
    margin: 15px 0 20px 0;
    flex-wrap: wrap;
}

.metric-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    padding: 12px 18px;
    min-width: 130px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(8px);
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #60a5fa;
    margin: 0;
}

.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
}

/* Source Badge Pills */
.source-pill {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 6px;
    margin-bottom: 6px;
}

.source-card {
    background: rgba(255, 255, 255, 0.02);
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    color: #cbd5e1;
}

/* Suggested Chips */
.suggestion-btn {
    margin-right: 8px;
    margin-bottom: 8px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 3. Prompt Template
SYSTEM_PROMPT = """You are an expert AI document assistant specializing in comprehensive and precise document analysis.
Answer the user's question accurately using ONLY the provided document context.

Guidelines:
1. Answer clearly, concisely, and structure with bullet points or sections where appropriate.
2. If the context does not contain sufficient information to answer the question, state: "The uploaded document(s) do not contain sufficient information to answer this question."
3. Do not invent or hallucinate information outside the retrieved context.

Context:
{context}
"""


# 4. Core Processing Functions
def extract_text_from_pdfs(uploaded_files) -> Tuple[List[Document], Dict[str, Any]]:
    documents = []
    file_stats = {}
    
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        page_count = len(reader.pages)
        file_stats[uploaded_file.name] = page_count
        
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
    return documents, file_stats


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


def format_chat_history(messages: List[Dict[str, Any]]) -> str:
    """Format past messages for conversational awareness."""
    formatted = []
    for msg in messages[-6:]:  # Last 3 conversational turns
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)


def stream_rag_response(
    query: str,
    vector_store,
    model_name: str,
    top_k: int,
    api_key: str
) -> Generator[str, None, List[Document]]:
    """Stream answer generator and return source documents."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    source_docs = retriever.invoke(query)
    
    # Format retrieved context
    context_blocks = []
    for doc in source_docs:
        header = f"[Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')} | Chunk: {doc.metadata.get('chunk_id')}]"
        context_blocks.append(f"{header}\n{doc.page_content}")
    context_str = "\n\n".join(context_blocks)
    
    # Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0.2,
        streaming=True,
        google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
    )
    
    chain = prompt | llm
    
    for chunk in chain.stream({"context": context_str, "question": query}):
        yield chunk.content, source_docs


# 5. Sidebar Setup & Controls
def sidebar():
    with st.sidebar:
        st.title("⚙️ Control Panel")
        
        # API Key Configuration
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            value=os.getenv("GOOGLE_API_KEY", ""),
            help="Get your free API key at aistudio.google.com"
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            
        st.markdown("---")
        st.subheader("🧠 Model & Retrieval")
        
        model_name = st.selectbox(
            "Gemini Model",
            ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-flash-latest"],
            index=0,
            help="gemini-2.5-flash is fast & accurate; gemini-2.5-pro is best for complex reasoning."
        )
        
        store_type = st.selectbox(
            "Vector Database",
            ["FAISS", "ChromaDB"],
            index=0,
            help="FAISS: In-memory & blazing fast. ChromaDB: Lightweight persistent database."
        )
        
        top_k = st.slider("Retrieved Chunks (k)", 2, 8, 4, 1, help="Number of document chunks to retrieve per question.")
        
        with st.expander("🛠️ Advanced Chunking Settings"):
            chunk_size = st.slider("Chunk Size", 300, 2000, 900, 100)
            chunk_overlap = st.slider("Chunk Overlap", 0, 400, 150, 10)

        # Document Stats Card in Sidebar
        if "stats" in st.session_state:
            st.markdown("---")
            st.subheader("📊 Document Insights")
            stats = st.session_state["stats"]
            st.caption(f"📁 Files: **{stats['files']}**")
            st.caption(f"📑 Pages: **{stats['pages']}**")
            st.caption(f"🧩 Chunks: **{stats['chunks']}**")
            st.caption(f"🗄️ Index: **{stats['store']}**")
            
        st.markdown("---")
        st.subheader("🧹 Session Controls")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()
        with col_s2:
            if st.button("Reset All", use_container_width=True):
                for key in ["vector_store", "chunks", "stats", "messages", "last_files"]:
                    st.session_state.pop(key, None)
                st.rerun()

        # Chat Export
        if st.session_state.get("messages"):
            st.markdown("---")
            chat_export = "\n\n".join(
                f"**{m['role'].upper()}:**\n{m['content']}" for m in st.session_state["messages"]
            )
            st.download_button(
                label="📥 Export Chat History",
                data=chat_export,
                file_name=f"chat_transcript_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    return api_key, model_name, store_type, top_k, chunk_size, chunk_overlap


# 6. Main Application Flow
def main():
    # Initialize Session States
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    api_key, model_name, store_type, top_k, chunk_size, chunk_overlap = sidebar()

    # Header
    st.markdown('<div class="main-header">📄 Multi-PDF RAG Assistant</div>', unsafe_allow_html=True)
    st.caption("Upload multiple PDFs, index semantic embeddings with FAISS/ChromaDB, and chat with your documents using Google Gemini.")

    # PDF Upload Section
    with st.expander("📤 Upload & Index PDF Documents", expanded=("vector_store" not in st.session_state)):
        uploaded_files = st.file_uploader(
            "Select one or more PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="You can drag & drop multiple research papers, reports, resumes, or manuals."
        )

        col_proc, _ = st.columns([1, 3])
        with col_proc:
            process_btn = st.button("🚀 Process & Build Index", use_container_width=True, type="primary")

        if process_btn:
            if not os.getenv("GOOGLE_API_KEY") and not api_key:
                st.error("⚠️ Please provide your Google Gemini API Key in the sidebar.")
            elif not uploaded_files:
                st.error("⚠️ Please upload at least one PDF file.")
            else:
                with st.spinner("Extracting text, generating embeddings, and building vector index..."):
                    raw_docs, file_stats = extract_text_from_pdfs(uploaded_files)
                    
                    if not raw_docs:
                        st.error("Could not extract any readable text from the uploaded PDF(s).")
                    else:
                        chunks = chunk_documents(raw_docs, chunk_size, chunk_overlap)
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model="models/gemini-embedding-001",
                            google_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
                        )
                        persist_dir = tempfile.mkdtemp(prefix="chromadb_")
                        vector_store = build_vector_store(chunks, embeddings, store_type, persist_dir)

                        # Save state
                        st.session_state["vector_store"] = vector_store
                        st.session_state["chunks"] = chunks
                        st.session_state["stats"] = {
                            "files": len(uploaded_files),
                            "pages": sum(file_stats.values()),
                            "chunks": len(chunks),
                            "store": store_type,
                            "model": model_name
                        }
                        st.success(f"✅ Successfully indexed {len(uploaded_files)} document(s) ({sum(file_stats.values())} pages, {len(chunks)} chunks) into {store_type}!")
                        st.rerun()

    # Document Metrics Display Banner
    if "stats" in st.session_state:
        stats = st.session_state["stats"]
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-card">
                    <p class="metric-value">{stats['files']}</p>
                    <p class="metric-label">Documents</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{stats['pages']}</p>
                    <p class="metric-label">Pages</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{stats['chunks']}</p>
                    <p class="metric-label">Chunks</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{stats['store']}</p>
                    <p class="metric-label">Vector Store</p>
                </div>
                <div class="metric-card">
                    <p class="metric-value">{model_name.replace('gemini-', '')}</p>
                    <p class="metric-label">Model</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Suggested Prompts / Quick Action Chips
    if "vector_store" in st.session_state:
        st.markdown("##### 💡 Suggested Questions")
        chip_cols = st.columns(4)
        suggested_queries = [
            ("📝 Summarize Document", "Please provide a comprehensive summary of the uploaded document(s), highlighting main topics and key takeaways."),
            ("🔑 Key Findings & Skills", "What are the key highlights, skills, or findings mentioned across the document(s)?"),
            ("❓ 3 Important Questions", "Generate 3 important questions that this document answers, along with concise answers based on the text."),
            ("📌 Key Data & Facts", "List all specific metrics, numbers, dates, or factual claims made in the document."),
        ]

        active_chip_query = None
        for i, (label, prompt_text) in enumerate(suggested_queries):
            with chip_cols[i]:
                if st.button(label, key=f"chip_{i}", use_container_width=True):
                    active_chip_query = prompt_text

    st.markdown("---")

    # Display Chat History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            
            # Display source citations accordion if available
            if msg.get("sources"):
                with st.expander(f"📚 View Cited Sources ({len(msg['sources'])} chunks)"):
                    for doc in msg["sources"]:
                        src = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "?")
                        cid = doc.metadata.get("chunk_id", "?")
                        st.markdown(
                            f'<span class="source-pill">📄 {src}</span><span class="source-pill">🔖 Page {page}</span><span class="source-pill">🧩 Chunk {cid}</span>',
                            unsafe_allow_html=True
                        )
                        st.markdown(f'<div class="source-card">{doc.page_content}</div>', unsafe_allow_html=True)

    # Chat Input Handler
    user_query = st.chat_input("Ask a question about your uploaded document(s)...")
    prompt_to_process = active_chip_query if ("vector_store" in st.session_state and active_chip_query) else user_query

    if prompt_to_process:
        if "vector_store" not in st.session_state:
            st.warning("⚠️ Please upload and process at least one PDF document first.")
        else:
            # Append & render user message
            st.session_state["messages"].append({"role": "user", "content": prompt_to_process})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt_to_process)

            # Assistant response container
            with st.chat_message("assistant", avatar="🤖"):
                response_placeholder = st.empty()
                full_response = ""
                collected_sources = []

                try:
                    # Stream generator
                    response_gen = stream_rag_response(
                        query=prompt_to_process,
                        vector_store=st.session_state["vector_store"],
                        model_name=model_name,
                        top_k=top_k,
                        api_key=api_key or os.getenv("GOOGLE_API_KEY")
                    )

                    for chunk_text, source_docs in response_gen:
                        full_response += chunk_text
                        collected_sources = source_docs
                        response_placeholder.markdown(full_response + "▌")
                    
                    response_placeholder.markdown(full_response)

                    # Display source citations accordion
                    if collected_sources:
                        with st.expander(f"📚 View Cited Sources ({len(collected_sources)} chunks)"):
                            for doc in collected_sources:
                                src = doc.metadata.get("source", "Unknown")
                                page = doc.metadata.get("page", "?")
                                cid = doc.metadata.get("chunk_id", "?")
                                st.markdown(
                                    f'<span class="source-pill">📄 {src}</span><span class="source-pill">🔖 Page {page}</span><span class="source-pill">🧩 Chunk {cid}</span>',
                                    unsafe_allow_html=True
                                )
                                st.markdown(f'<div class="source-card">{doc.page_content}</div>', unsafe_allow_html=True)

                    # Store assistant message in history
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": full_response,
                        "sources": collected_sources
                    })

                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")


if __name__ == "__main__":
    main()
