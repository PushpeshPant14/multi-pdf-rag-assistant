# 📄 Multi-PDF RAG Assistant

An interactive, production-ready Streamlit application that allows users to upload multiple PDF documents and chat with them using **Retrieval-Augmented Generation (RAG)** powered by **Google Gemini** and **LangChain**.

---

## 🌟 Key Features

- **📑 Multi-PDF Upload & Extraction**: Reads and parses text across multiple PDF documents while retaining page-level metadata.
- **✂️ Recursive Document Chunking**: Splits large texts into semantically coherent segments with configurable chunk size and overlap.
- **⚡ Dual Vector Store Support**:
  - **FAISS**: In-memory, ultra-fast vector similarity search.
  - **ChromaDB**: Lightweight, persistent vector database.
- **🧠 Modern Google Gemini Integration**:
  - **Embeddings**: Uses `models/gemini-embedding-001` (3072-dimensional vector space).
  - **LLM**: Supports `gemini-2.5-flash`, `gemini-2.5-pro`, and `gemini-flash-latest`.
- **🎯 Hallucination-Resistant & Grounded**: Constrains answers strictly to the retrieved context.
- **🔍 Source Citations**: Every response includes file names, page numbers, and chunk references used to construct the answer.
- **⚙️ Dynamic Configuration**: Easily tune chunk parameters, switch vector databases, and select Gemini models via the sidebar UI.

---

## 🏗️ Architecture & Pipeline

```
[ User Uploads PDFs ]
         │
         ▼
[ PDF Text Extraction (pypdf) ] ──▶ [ Page Metadata Tagging ]
         │
         ▼
[ Recursive Character Chunking ]
         │
         ▼
[ Gemini Embeddings Generation ]
         │
         ▼
[ Vector Store Indexing (FAISS / ChromaDB) ]
         │
         ├─── [ User Asks a Question ]
         │              │
         ▼              ▼
[ Similarity Search (Top-k Chunks) ]
         │
         ▼
[ Grounded QA Prompt with Retrieved Context ]
         │
         ▼
[ Gemini 2.5 LLM Answer + Page/Chunk Source Citations ]
```

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Orchestration**: [LangChain](https://www.langchain.com/) (`langchain-classic`, `langchain-core`, `langchain-google-genai`)
- **LLM & Embeddings**: [Google Gemini API](https://ai.google.dev/)
- **Vector Stores**: [FAISS](https://github.com/facebookresearch/faiss), [ChromaDB](https://www.trychroma.com/)
- **PDF Parser**: [pypdf](https://pypdf.readthedocs.io/)
- **Environment Management**: `python-dotenv`

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/PushpeshPant14/multi-pdf-rag-assistant.git
cd multi-pdf-rag-assistant
```

### 2. Configure Your API Key
Create a `.env` file in the root directory:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```
> 🔑 Get your free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

---

### 3. Run the Application

#### ⚡ Option 1: Using `uv` (Recommended — 1 Command Setup)

If `uv` is not installed yet, install it quickly:
- **Windows (PowerShell)**: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **macOS / Linux**: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Or via pip**: `pip install uv`

Then simply run:
```bash
uv run streamlit run app.py
```
> **Why `uv`?** `uv` automatically creates a virtual environment, resolves all dependencies from `pyproject.toml`, installs them in seconds, and launches the app — no manual `pip install` or environment activation required!

---

#### 🐍 Option 2: Using Standard Python `venv` + `pip`

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```text
multi-pdf-rag-assistant/
│
├── app.py              # Main Streamlit application & RAG pipeline
├── requirements.txt    # Python dependencies (for pip)
├── pyproject.toml      # Project configuration (for uv / poetry)
├── .env                # Gemini API key configuration (keep secret)
├── .gitignore          # Git ignore file
└── README.md           # Project documentation
```

---

## 💡 How to Use

1. **Enter API Key**: Provide your Google Gemini API Key in the sidebar (or load via `.env`).
2. **Configure Settings**: Select your preferred Gemini Model, Vector Store (FAISS or ChromaDB), and chunk size.
3. **Upload PDFs**: Use the file uploader to drag-and-drop one or more PDF files.
4. **Process Documents**: Click **"Process Documents"** to extract text, generate embeddings, and build the vector index.
5. **Ask Questions**: Type your question in the search box and click **"Get Answer"** to receive a grounded answer with cited sources.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
