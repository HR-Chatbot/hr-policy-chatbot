"""
HR Policy Chatbot for Indian Companies
Features:
- Reads PDF policies from /policies folder
- Semantic search for relevant policy sections
- Conversation memory
- Gemini AI for follow-up questions with Indian HR context
- 24/7 availability on Streamlit Cloud
"""

import streamlit as st
import os
import re
from pathlib import Path
from google import genai
from PyPDF2 import PdfReader
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import time

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
    .policy-source {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .contact-hr {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-top: 1rem;
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============== INITIALIZATION ==============
def init_session_state():
    """Initialize session state variables"""
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
    if 'genai_client' not in st.session_state:
        st.session_state.genai_client = None

# ============== PDF PROCESSING ==============
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading {pdf_path}: {str(e)}")
        return ""

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks for better context"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:  # Only keep substantial chunks
            chunks.append(chunk)
    
    return chunks

def load_policies():
    """Load all PDF policies from the policies folder"""
    policies_dir = Path("policies")
    
    if not policies_dir.exists():
        st.error("❌ Policies folder not found!")
        return [], [], []
    
    pdf_files = list(policies_dir.glob("*.pdf"))
    
    if not pdf_files:
        st.warning("⚠️ No PDF files found in policies folder")
        return [], [], []
    
    all_chunks = []
    chunk_sources = []
    all_texts = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_file in enumerate(pdf_files):
        status_text.text(f"📄 Loading: {pdf_file.name}...")
        progress = (idx + 1) / len(pdf_files)
        progress_bar.progress(progress)
        
        text = extract_text_from_pdf(pdf_file)
        if text:
            all_texts.append(text)
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_sources.append(pdf_file.name)
    
    progress_bar.empty()
    status_text.empty()
    
    return all_chunks, chunk_sources, pdf_files

# ============== SEARCH FUNCTIONALITY ==============
def setup_vectorizer(chunks):
    """Setup TF-IDF vectorizer for semantic search"""
    if not chunks:
        return None, None
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def find_relevant_chunks(query, top_k=3):
    """Find most relevant policy chunks for a query"""
    if st.session_state.vectorizer is None or not st.session_state.policy_chunks:
        return [], []
    
    query_vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
    
    # Get top-k indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    relevant_chunks = [st.session_state.policy_chunks[i] for i in top_indices]
    scores = [similarities[i] for i in top_indices]
    
    return relevant_chunks, scores

# ============== GEMINI AI SETUP ==============
def setup_gemini():
    """Setup Gemini API using new google-genai package"""
    try:
        # Try to get API key from Streamlit secrets (for deployment)
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except:
        # Fallback to environment variable
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
    """Get response from Gemini with Indian HR context"""
    try:
        # Build conversation history
        history_text = ""
        for msg in chat_history[-3:]:  # Last 3 messages for context
            role = "Employee" if msg['role'] == 'user' else "HR Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        prompt = f"""You are an HR Policy Assistant for an Indian company. Your role is to help employees understand HR policies and procedures.

CONTEXT FROM COMPANY POLICY DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history_text}

EMPLOYEE QUESTION: {query}

INSTRUCTIONS:
1. If the context contains relevant information, answer based on the policy documents first.
2. If the employee asks follow-up questions or needs general HR guidance relevant to Indian companies, provide helpful information based on standard Indian HR practices.
3. Keep answers concise, professional, and friendly.
4. If you don't know the answer or it's not in the policies, say "I don't have specific information about this in our policy documents. Please contact the HR department for assistance."
5. For leave-related questions, mention that they need to apply through the official HR portal or contact their manager.
6. Always maintain a helpful, professional tone suitable for workplace communication.

Provide a clear, accurate response:"""
        
        # Use the new API format
        response = client.models.generate_content(
            model="gemini-1.5-flash-002",",
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        return f"I apologize, but I'm having trouble processing your request. Please contact HR directly for assistance. Error: {str(e)}"

# ============== MAIN CHATBOT LOGIC ==============
def process_query(query):
    """Process user query and generate response"""
    
    # Find relevant policy chunks
    relevant_chunks, scores = find_relevant_chunks(query)
    
    # Build context from relevant chunks
    context = "\n\n".join([f"[Relevance: {score:.2f}]\n{chunk}" 
                          for chunk, score in zip(relevant_chunks, scores) if score > 0.1])
    
    # If no relevant context found, use generic message
    if not context:
        context = "No specific policy information found for this query."
    
    # Get AI response
    client = st.session_state.genai_client
    if client:
        response = get_gemini_response(query, context, st.session_state.chat_history, client)
    else:
        response = "I'm currently operating in policy search mode only. For AI-powered answers, please ensure the API key is configured."
    
    # Determine if we should show "Contact HR" suggestion
    show_contact_hr = any(phrase in response.lower() for phrase in [
        "don't have specific information",
        "contact hr",
        "not found in policy",
        "i don't know"
    ])
    
    return response, relevant_chunks, show_contact_hr

# ============== UI COMPONENTS ==============
def display_chat_history():
    """Display chat history"""
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <b>👤 You:</b><br>{message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="chat-message bot-message">
                    <b>🤖 HR Assistant:</b><br>{message['content']}
                </div>
            """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<div class="main-header">💼 HR Policy Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Your 24/7 HR companion. Ask about leave, attendance, benefits, and more.</div>', unsafe_allow_html=True)
    
    # Initialize
    init_session_state()
    
    # Setup Gemini client if not already done
    if st.session_state.genai_client is None:
        st.session_state.genai_client = setup_gemini()
    
    # Load policies (only once)
    if not st.session_state.policies_loaded:
        with st.spinner("📚 Loading HR policies..."):
            chunks, sources, pdf_files = load_policies()
            
            if chunks:
                st.session_state.policy_chunks = chunks
                st.session_state.policy_sources = sources
                st.session_state.vectorizer, st.session_state.tfidf_matrix = setup_vectorizer(chunks)
                st.session_state.policies_loaded = True
                st.success(f"✅ Loaded {len(pdf_files)} policy documents")
            else:
                st.error("❌ No policies loaded. Please upload PDF files to the policies folder.")
                return
    
    # Check for API key
    if st.session_state.genai_client is None:
        st.warning("⚠️ Gemini API key not configured. The chatbot will use policy search only.")
        st.info("To enable AI responses, add your GEMINI_API_KEY to Streamlit secrets.")
    
    # Display chat history
    display_chat_history()
    
    # Input area
    st.markdown("---")
    query = st.text_input("Ask your HR question:", 
                         placeholder="E.g., How many casual leaves do I have? How do I apply for leave?",
                         key="user_input")
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        submit = st.button("🚀 Ask", use_container_width=True)
    
    with col2:
        clear = st.button("🔄 Clear Chat", use_container_width=True)
    
    if clear:
        st.session_state.chat_history = []
        st.rerun()
    
    if submit and query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": query})
        
        # Process query
        with st.spinner("🤔 Thinking..."):
            response, relevant_chunks, show_contact_hr = process_query(query)
        
        # Add bot response to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to display updated chat
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.9rem;">
            <p>🕐 Available 24/7 | 🔒 Your conversations are not stored permanently</p>
            <p>⚠️ For complex issues, please contact HR directly</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
