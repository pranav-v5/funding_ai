import streamlit as st
import os

# Delay heavy imports to let Streamlit UI render first
# from langchain_community.document_loaders import TextLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.docstore.document import Document

# --- Configuration ---
st.set_page_config(page_title="Funding AI", page_icon="💰", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        background-color: #262730;
        color: #ffffff;
    }
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .grant-card {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 10px;
    }
    .source-tag {
        background-color: #333;
        padding: 2px 8px;
        border-radius: 5px;
        font-size: 12px;
        margin-right: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💰 Funding AI Intelligence Platform")
st.markdown("### *Bridging Structured Data & Unstructured Semantic Search for GSoC 2026*")

# API Keys
groq_api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

# --- Initialize Resources ---
@st.cache_resource
def get_vector_db():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.docstore.document import Document

    st.info("Loading funding documents and creating vector index... (This may take a minute on first run)")
    print("DEBUG: Starting get_vector_db...")
    # Mock documents with simple structured metadata
    docs_info = [
        {
            "file": "data/ai_grant_2026.txt",
            "metadata": {
                "title": "AI Research Grant 2026",
                "category": "AI",
                "deadline": "May 15, 2026",
                "eligibility": "PhD/Master Students"
            }
        },
        {
            "file": "data/climate_fund_2026.txt",
            "metadata": {
                "title": "Climate Innovation Fund",
                "category": "Climate",
                "deadline": "Oct 30, 2026",
                "eligibility": "Startups/Research Labs"
            }
        },
        {
            "file": "data/tech_scholarship_2027.txt",
            "metadata": {
                "title": "Global Tech Scholarship 2027",
                "category": "Tech",
                "deadline": "Jan 20, 2027",
                "eligibility": "Undergraduate Students"
            }
        }
    ]

    all_docs = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    print("DEBUG: Splitting texts...")
    for item in docs_info:
        if os.path.exists(item["file"]):
            with open(item["file"], "r", encoding="utf-8") as f:
                content = f.read()
                chunks = text_splitter.split_text(content)
                for chunk in chunks:
                    all_docs.append(Document(page_content=chunk, metadata=item["metadata"]))

    print("DEBUG: Downloading embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("DEBUG: Creating Chroma index...")
    vectorstore = Chroma.from_documents(all_docs, embeddings, persist_directory="./chroma_db")
    print("DEBUG: Vector DB ready.")
    return vectorstore

# --- Main App ---
if groq_api_key:
    from langchain_groq import ChatGroq
    db = get_vector_db()
    llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.1-8b-instant", temperature=0.3)
    
    query = st.text_input("Search for grants (e.g., 'AI grants for students'):")
    
    if query:
        with st.spinner("Processing query..."):
            # Retrieval
            relevant_docs = db.similarity_search(query, k=3)
            
            # Format Context
            context = "\n\n".join([f"Source: {d.metadata['title']}\nContent: {d.page_content}" for d in relevant_docs])
            
            prompt = f"""
            You are a funding intelligence assistant. Use the following context to answer the user query.
            If the answer is not in the context, say you don't know based on the provided documents.
            
            CONTEXT:
            {context}
            
            QUERY:
            {query}
            
            Format your response clearly. 
            Include an 'Answer' section followed by a 'Relevant Funding' list with metadata (Title, Deadline, Eligibility).
            """
            
            response = llm.invoke(prompt)
            
            # Display Results
            st.divider()
            st.markdown(f"### Answer:")
            st.write(response.content)
            
            st.markdown("### 🧩 Relevant Funding Details (Metadata Extraction):")
            for doc in relevant_docs:
                meta = doc.metadata
                st.markdown(f"""
                <div class="grant-card">
                    <b>{meta['title']}</b><br>
                    📅 Deadline: {meta['deadline']} | 🎓 Eligibility: {meta['eligibility']} | 🏷️ Category: {meta['category']}
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📚 Source Documents:")
            seen_sources = set()
            for doc in relevant_docs:
                source = doc.metadata['title']
                if source not in seen_sources:
                    st.markdown(f'<span class="source-tag">📄 {source}</span>', unsafe_allow_html=True)
                    seen_sources.add(source)
else:
    st.info("👈 Please enter your Groq API Key in the sidebar to begin.")

# Project Showcase Sidebar
st.sidebar.divider()
st.sidebar.markdown("""
### 🧠 Architecture
1. **Documents**: Funding PDFs/Text
2. **Chunking**: Semantic splitting
3. **Embeddings**: Sentence-BERT (L6-v2)
4. **Vector DB**: Chroma (with Metadata)
5. **Retriever**: Top-K Semantic Search
6. **LLM**: Llama 3.1 8B (Groq)
""")
