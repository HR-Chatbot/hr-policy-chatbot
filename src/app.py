"""
HR Policy Chatbot for Indian Companies
Uses google-genai package with gemini-2.5-flash model
"""

import streamlit as st
import os
from pathlib import Path
from google import genai
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============== CONFIGURATION ==============
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left: 5px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

# ============== INITIALIZATION ==============
def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'policy_chunks' not in st.session_state:
        st.session_state.policy_chunks = []
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None
    if 'tfidf_matrix' not in st.session_state:
        st.session_state.tfidf_matrix = None
    if 'policies_loaded' not in st.session_state:
        st.session_state.policies_loaded = False

# ============== PDF PROCESSING ==============
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return ""

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def load_policies():
    policies_dir = Path("policies")
    if not policies_dir.exists():
        return [], [], []
    
    pdf_files = list(policies_dir.glob("*.pdf"))
    all_chunks = []
    chunk_sources = []
    
    progress_bar = st.progress(0)
    for idx, pdf_file in enumerate(pdf_files):
        progress = (idx + 1) / len(pdf_files)
        progress_bar.progress(progress)
        
        text = extract_text_from_pdf(pdf_file)
        if text:
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_sources.append(pdf_file.name)
    
    progress_bar.empty()
    return all_chunks, chunk_sources, pdf_files

# ============== SEARCH FUNCTIONALITY ==============
def setup_vectorizer(chunks):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def find_relevant_chunks(query, top_k=3):
    if st.session_state.vectorizer is None:
        return [], []
    query_vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [st.session_state.policy_chunks[i] for i in top_indices]
    return relevant_chunks, similarities[top_indices]

# ============== GEMINI AI SETUP ==============
def setup_gemini():
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini: {str(e)}")
        return None

def get_gemini_response(query, context, chat_history, client):
    try:
        history_text = ""
        for msg in chat_history[-3:]:
            role = "Employee" if msg['role'] == 'user' else "HR Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        prompt = f"""You are an HR Policy Assistant for an Indian company.

CONTEXT FROM POLICY DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history_text}

QUESTION: {query}

Provide a helpful, professional response based on the context. If you do not know the answer, say 'Please contact HR directly for assistance.'"""
        
        # Use gemini-2.5-flash model (you have 20 requests/day available)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}"

# ============== MAIN CHATBOT LOGIC ==============
def process_query(query, client):
    relevant_chunks, scores = find_relevant_chunks(query)
    context = "\n\n".join([f"{chunk}" for chunk, score in zip(relevant_chunks, scores) if score > 0.1])
    
    if not context:
        context = "No specific policy information found."
    
    if client:
        response = get_gemini_response(query, context, st.session_state.chat_history, client)
    else:
        response = "API not configured. Please contact HR."
    
    return response

# ============== UI ==============
def display_chat_history():
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f'<div class="chat-message user-message"><b>👤 You:</b><br>{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message"><b>🤖 HR Assistant:</b><br>{message["content"]}</div>', unsafe_allow_html=True)

def main():
    st.markdown('<div class="main-header">💼 HR Policy Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your 24/7 HR companion</div>', unsafe_allow_html=True)
    
    init_session_state()
    
    client = setup_gemini()
    
    if not st.session_state.policies_loaded:
        with st.spinner("📚 Loading policies..."):
            chunks, sources, pdf_files = load_policies()
            if chunks:
                st.session_state.policy_chunks = chunks
                st.session_state.vectorizer, st.session_state.tfidf_matrix = setup_vectorizer(chunks)
                st.session_state.policies_loaded = True
                st.success(f"✅ Loaded {len(pdf_files)} policy documents")
            else:
                st.error("❌ No policies loaded")
                return
    
    if client is None:
        st.warning("⚠️ API key not configured")
    
    display_chat_history()
    
    st.markdown("---")
    query = st.text_input("Ask your HR question:", placeholder="E.g., How many casual leaves do I have?", key="user_input")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.button("🚀 Ask", use_container_width=True)
    with col2:
        clear = st.button("🔄 Clear Chat")
    
    if clear:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit and query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.spinner("🤔 Thinking..."):
            response = process_query(query, client)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
    
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: #666; font-size: 0.9rem;"><p>🕐 Available 24/7 | For complex issues, contact HR directly</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
