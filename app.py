import os
import numpy as np
import faiss
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIGURATION
# -----------------------------------------------------------------------------
GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 3

st.set_page_config(
    page_title="RAG Document Chatbot",
    page_icon="📚",
    layout="wide"
)

# -----------------------------------------------------------------------------
# SECURE API KEY RETRIEVAL
# -----------------------------------------------------------------------------
def get_groq_api_key():
    """Retrieve Groq API key from Streamlit secrets or OS Environment Variables."""
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY", "")

# -----------------------------------------------------------------------------
# EMBEDDING MODEL LOADING
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load and cache the sentence transformer model in memory."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

# -----------------------------------------------------------------------------
# DOCUMENT EXTRACTION
# -----------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF file and record page metadata."""
    documents = []
    scanned_warning = False
    try:
        reader = PdfReader(uploaded_file)
        has_text = False
        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                has_text = True
                documents.append({
                    "file_name": uploaded_file.name,
                    "page_number": idx,
                    "text": text
                })
        if len(reader.pages) > 0 and not has_text:
            scanned_warning = True
    except Exception as e:
        st.error(f"Error processing PDF '{uploaded_file.name}': {str(e)}")
    return documents, scanned_warning

def extract_text_from_txt(uploaded_file):
    """Extract text from uploaded TXT file."""
    documents = []
    try:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        if content.strip():
            documents.append({
                "file_name": uploaded_file.name,
                "page_number": 1,
                "text": content
            })
    except Exception as e:
        st.error(f"Error reading TXT file '{uploaded_file.name}': {str(e)}")
    return documents

# -----------------------------------------------------------------------------
# TEXT CHUNKING
# -----------------------------------------------------------------------------
def chunk_documents(documents, chunk_size, chunk_overlap):
    """Split extracted text documents into overlapping chunks with metadata."""
    chunks = []
    counter = 0
    for doc in documents:
        text = doc["text"]
        start = 0
        text_len = len(text)
        while start < text_len:
            end = start + chunk_size
            segment = text[start:end]
            if segment.strip():
                counter += 1
                chunks.append({
                    "chunk_id": counter,
                    "file_name": doc["file_name"],
                    "page_number": doc["page_number"],
                    "text": segment.strip()
                })
            start += (chunk_size - chunk_overlap)
    return chunks

# -----------------------------------------------------------------------------
# VECTOR DATABASE (FAISS)
# -----------------------------------------------------------------------------
def build_vector_store(chunks, model):
    """Generate embeddings for text chunks and populate FAISS index."""
    if not chunks:
        return None, None
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, chunks

def query_vector_store(query, index, chunks, model, top_k):
    """Search FAISS vector store for top matching document chunks."""
    if index is None or not chunks:
        return []
    query_vector = model.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)
    distances, indices = index.search(query_vector, top_k)
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(chunks):
            results.append(chunks[idx])
    return results

# -----------------------------------------------------------------------------
# GROQ LLM GENERATION
# -----------------------------------------------------------------------------
def generate_rag_response(api_key, model_name, query, context_chunks, chat_history):
    """Formulate grounded prompt and query Groq API."""
    if not api_key:
        return "Error: Groq API Key missing. Please provide it in Secrets or the sidebar.", []
    
    client = Groq(api_key=api_key)
    context_text = ""
    for idx, chunk in enumerate(context_chunks, start=1):
        context_text += f"\n--- SOURCE BLOCK {idx} [File: {chunk['file_name']}, Page: {chunk['page_number']}] ---\n{chunk['text']}\n"

    system_prompt = (
        "You are an AI assistant answering questions strictly based on the provided document context.\n"
        "STRICT RULES:\n"
        "1. Answer strictly using only the provided document context.\n"
        "2. Do NOT use outside knowledge or make assumptions.\n"
        "3. If the answer cannot be determined strictly from the context, state EXACTLY:\n"
        "   'I could not find this information in the uploaded documents.'\n"
        "4. Cite source document names and page numbers where applicable."
    )

    user_prompt = f"PROVIDED CONTEXT:\n{context_text}\n\nUSER QUESTION: {query}"
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-4:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=800
        )
        return response.choices[0].message.content, context_chunks
    except Exception as e:
        return f"Groq API Error: {str(e)}", []

# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "kb_processed" not in st.session_state:
    st.session_state.kb_processed = False
if "uploaded_filenames" not in st.session_state:
    st.session_state.uploaded_filenames = []

embedding_model = load_embedding_model()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & CONTROLS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ RAG Settings")
    active_api_key = get_groq_api_key()
    if not active_api_key:
        active_api_key = st.text_input("Enter Groq API Key:", type="password")
        if active_api_key:
            st.success("API Key configured!")
    else:
        st.success("Groq API Key detected.", icon="✅")

    st.divider()
    selected_model = st.selectbox(
        "Groq Model:",
        options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile"],
        index=0
    )

    st.divider()
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files:",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    with st.expander("Advanced Hyperparameters"):
        chunk_size = st.slider("Chunk Size", 300, 2000, DEFAULT_CHUNK_SIZE, step=50)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, DEFAULT_CHUNK_OVERLAP, step=10)
        top_k = st.slider("Retrieved Chunks (Top K)", 1, 10, DEFAULT_TOP_K)

    process_btn = st.button("🚀 Process Documents", use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset KB", use_container_width=True):
            st.session_state.faiss_index = None
            st.session_state.doc_chunks = []
            st.session_state.kb_processed = False
            st.session_state.uploaded_filenames = []
            st.session_state.messages = []
            st.rerun()

# -----------------------------------------------------------------------------
# DOCUMENT PROCESSING PIPELINE
# -----------------------------------------------------------------------------
if process_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF or TXT file.")
    else:
        with st.status("Processing documents...", expanded=True) as status:
            all_documents = []
            scanned_files = []
            for file in uploaded_files:
                if file.name.lower().endswith(".pdf"):
                    docs, is_scanned = extract_text_from_pdf(file)
                    all_documents.extend(docs)
                    if is_scanned:
                        scanned_files.append(file.name)
                elif file.name.lower().endswith(".txt"):
                    docs = extract_text_from_txt(file)
                    all_documents.extend(docs)

            if scanned_files:
                st.warning(f"Warning: Scanned PDF detected (no extractable text): {', '.join(scanned_files)}")

            if not all_documents:
                status.update(label="No readable text found in documents.", state="error")
            else:
                st.write("Chunking document text...")
                chunks = chunk_documents(all_documents, chunk_size, chunk_overlap)
                st.write("Generating vector store...")
                index, valid_chunks = build_vector_store(chunks, embedding_model)

                st.session_state.faiss_index = index
                st.session_state.doc_chunks = valid_chunks
                st.session_state.kb_processed = True
                st.session_state.uploaded_filenames = [f.name for f in uploaded_files]

                status.update(label="Knowledge Base created successfully!", state="complete")
                st.success(f"Indexed {len(valid_chunks)} chunks from {len(uploaded_files)} document(s).")

# -----------------------------------------------------------------------------
# MAIN CHAT APPLICATION INTERFACE
# -----------------------------------------------------------------------------
st.title("📚 RAG Document Chatbot")
st.caption("Upload PDF or TXT files in the sidebar, click Process, and ask grounded questions.")

if st.session_state.kb_processed:
    st.info(f"🟢 Knowledge Base Active: **{', '.join(st.session_state.uploaded_filenames)}** ({len(st.session_state.doc_chunks)} chunks indexed)")
else:
    st.warning("⚠️ Upload files in the sidebar and click 'Process Documents' to get started.")

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for s in message["sources"]:
                    st.markdown(f"**{s['file_name']}** (Page {s['page_number']}) — *Chunk #{s['chunk_id']}*")
                    st.caption(f"\"{s['text']}\"")

# Handle User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not active_api_key:
        st.error("Please configure your Groq API Key in the sidebar or Secrets.")
    elif not st.session_state.kb_processed:
        st.error("Please upload and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents & generating answer..."):
                retrieved_chunks = query_vector_store(
                    prompt,
                    st.session_state.faiss_index,
                    st.session_state.doc_chunks,
                    embedding_model,
                    top_k
                )
                answer, sources = generate_rag_response(
                    active_api_key,
                    selected_model,
                    prompt,
                    retrieved_chunks,
                    st.session_state.messages
                )
                st.markdown(answer)

                unique_sources = []
                seen = set()
                for s in sources:
                    pair = (s["file_name"], s["page_number"])
                    if pair not in seen:
                        seen.add(pair)
                        unique_sources.append(s)

                if unique_sources:
                    with st.expander("🔍 View Retrieved Sources"):
                        for s in unique_sources:
                            st.markdown(f"**{s['file_name']}** (Page {s['page_number']}) — *Chunk #{s['chunk_id']}*")
                            st.caption(f"\"{s['text']}\"")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": unique_sources
        })
