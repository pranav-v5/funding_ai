# 💰 Funding AI Intelligence Platform

A RAG-powered system for intelligent funding discovery, built for GSoC 2026.

## 🚀 Overview
**Funding AI** is a retrieval-augmented generation (RAG) platform that processes funding documents (Grants, Scholarships, Research Funds) to provide structured answers. Unlike basic RAG, this system integrates **Simple Metadata Extraction** to show structured info like deadlines and eligibility alongside natural language answers.

## 🛠 Tech Stack
- **Framework**: LangChain, Streamlit
- **LLM**: Llama 3.1 8B (Groq)
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence-BERT (MiniLM-L6-v2)
- **Deployment**: Readily hostable on Streamlit Cloud

## 🧩 Architecture
- **Inbound Documents**: Funding descriptions (.txt, .pdf ready)
- **Chunking**: RecursiveCharacterTextSplitter for semantic relevance.
- **Metadata**: Structured (Title, Deadline, Eligibility, Category) stored in Vector Store.
- **Search**: Hybrid context/metadata retrieval.
- **Output**: Structured responses with verifiable source tags.

## 🏃 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run app: `streamlit run app.py`
3. Enter your **Groq API Key** and start searching!

## 🧠 Sample Queries
- "What AI grants are available for students?"
- "Find climate funding with deadlines in 2026."
- "Scholarships for undergraduate tech students."
