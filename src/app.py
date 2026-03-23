"""
HR Policy Chatbot for Spectron - Suggestive Response Version
Analyzes policies and provides specific suggestions, not generic welcomes
"""

import streamlit as st
import os
import re
from pathlib import Path
from openai import OpenAI
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="Spectron HR Assistant | 24/7 Support",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 100% !important;
    }
    
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        margin-bottom: 0.3rem;
        margin-top: 0.5rem;
    }
    
    .sub-header {
        font-size: 1rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
    }
    
    .welcome-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .welcome-text {
        text-align: center;
        font-size: 0.95rem;
        line-height: 1.5;
        opacity: 0.95;
    }
    
    .policy-links-section {
        margin-bottom: 1rem;
    }
    
    .policy-links-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .policy-list-container {
        background: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.5rem;
        max-height: 200px;
        overflow-y: auto;
    }
    
    .policy-list-container::-webkit-scrollbar {
        width: 6px;
    }
    
    .policy-list-container::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    .policy-list-container::-webkit-scrollbar-thumb {
        background: #c53030;
        border-radius: 4px;
    }
    
    .policy-item {
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.9rem;
        color: #2d3748;
        transition: background 0.2s ease;
    }
    
    .policy-item:hover {
        background: #edf2f7;
    }
    
    .policy-item:last-child {
        border-bottom: none;
    }
    
    .chat-container {
        background: #f7fafc;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        min-height: 150px;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .chat-message {
        padding: 0.8rem 1rem;
        border-radius: 16px;
        margin-bottom: 0.75rem;
        max-width: 90%;
        animation: fadeIn 0.3s ease;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .user-message {
        background: linear-gradient(135deg, #c53030 0%, #9b2c2c 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        box-shadow: 0 3px 10px rgba(197, 48, 48, 0.3);
    }
    
    .bot-message {
        background: white;
        color: #2d3748;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    .message-header {
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        opacity: 0.9;
    }
    
    .input-area {
        background: white;
        padding: 0.75rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 0.5rem;
    }
    
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem;
        font-size: 0.95rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #c53030;
        box-shadow: 0 0 0 2px rgba(197, 48, 48, 0.1);
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.8rem 1rem;
        background: white;
        border-radius: 16px;
        border-bottom-left-radius: 4px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.08);
        width: fit-content;
        margin-bottom: 0.75rem;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        background: #c53030;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-8px); }
    }
    
    .policy-answer {
        line-height: 1.7;
    }
    
    .policy-answer strong {
        color: #1a365d;
    }
    
    .suggestion-box {
        background: #e6fffa;
        border-left: 3px solid #38b2ac;
        padding: 0.75rem;
        border-radius: 6px;
        margin-bottom: 0.75rem;
    }
    
    .suggestion-title {
        font-weight: 600;
        color: #234e52;
        margin-bottom: 0.3rem;
    }
    
    .procedure-box {
        background: #f0fff4;
        border-left: 3px solid #48bb78;
        padding: 0.75rem;
        border-radius: 6px;
        margin-top: 0.75rem;
    }
    
    .procedure-title {
        font-weight: 600;
        color: #22543d;
        margin-bottom: 0.3rem;
    }
    
    .decline-box {
        background: #fff5f5;
        border-left: 3px solid #c53030;
        padding: 0.75rem;
        border-radius: 6px;
        color: #c53030;
        font-size: 0.9rem;
    }
    
    .contact-card {
        background: linear-gradient(135deg, #f6e05e 0%, #d69e2e 100%);
        color: #744210;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    
    .contact-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    /* Hide ALL unwanted elements */
    .stDeployButton, #MainMenu, footer, header, .stSpinner,
    [data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    
    @media (max-width: 768px) {
        .main-header { font-size: 1.5rem; }
        .chat-message { max-width: 95%; font-size: 0.9rem; }
        .welcome-card { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# ============== INITIALIZATION ==============
def init_session_state():
    defaults = {
        'chat_history': [],
        'policy_chunks': [],
        'policy_sources': [],
        'vectorizer': None,
        'tfidf_matrix': None,
        'policies_loaded': False,
        'openai_client': None,
        'show_typing': False,
        'policy_texts': {},
        'policy_files': [],
        'input_counter': 0,
        'last_query': ''
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============== PDF PROCESSING ==============
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except:
        return ""

def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 40:
            chunks.append(chunk)
    return chunks

def load_policies():
    policies_dir = Path("policies")
    if not policies_dir.exists():
        return [], [], {}, []
    
    pdf_files = list(policies_dir.glob("*.pdf"))
    all_chunks, chunk_sources = [], []
    policy_texts = {}
    
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            policy_texts[pdf_file.name] = text
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_sources.append(pdf_file.name)
    
    return all_chunks, chunk_sources, policy_texts, pdf_files

def setup_vectorizer(chunks):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def search_policy(query, top_k=5):
    """Search for relevant policy chunks"""
    if st.session_state.vectorizer is None:
        return [], []
    
    query_vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    chunks = [st.session_state.policy_chunks[i] for i in top_indices]
    scores = [similarities[i] for i in top_indices]
    
    return chunks, scores

# ============== OPENAI SETUP ==============
def setup_openai():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        return OpenAI(api_key=api_key)
    except:
        return None

def is_greeting(text):
    """Check if text is a greeting"""
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'namaste', 'hola', 'greetings']
    text_lower = text.lower().strip()
    return any(g in text_lower for g in greetings) or len(text_lower) < 3

def find_matching_policy_file(query):
    """Find the best matching policy file based on query"""
    query_lower = query.lower()
    query_normalized = query_lower.replace(' ', '').replace('-', '').replace('_', '')
    query_words = set(w for w in query_lower.split() if len(w) > 2)
    
    best_match = None
    best_score = 0
    
    for pdf_file in st.session_state.policy_files:
        filename = pdf_file.name.lower().replace('.pdf', '')
        filename_normalized = filename.replace(' ', '').replace('-', '').replace('_', '')
        
        score = 0
        
        # Check for exact substring match
        if query_normalized in filename_normalized:
            score += 10
        
        # Check for word matches in filename
        filename_words = set(filename.split())
        matching_words = query_words.intersection(filename_words)
        score += len(matching_words) * 3
        
        # Check for partial word matches
        for qw in query_words:
            if qw in filename_normalized:
                score += 2
        
        # Check reverse: filename words in query
        for fw in filename_words:
            if fw in query_normalized and len(fw) > 3:
                score += 1
        
        if score > best_score:
            best_score = score
            best_match = pdf_file.name
    
    return best_match if best_score >= 2 else None

def get_chunks_from_specific_policy(policy_filename, query, top_k=5):
    """Get chunks specifically from the identified policy file"""
    specific_chunks = []
    
    for i, source in enumerate(st.session_state.policy_sources):
        if source == policy_filename:
            specific_chunks.append(st.session_state.policy_chunks[i])
    
    if not specific_chunks:
        return [], []
    
    if st.session_state.vectorizer:
        temp_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        temp_matrix = temp_vectorizer.fit_transform(specific_chunks)
        query_vec = temp_vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, temp_matrix).flatten()
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        ranked_chunks = [specific_chunks[i] for i in top_indices]
        scores = [similarities[i] for i in top_indices]
        
        return ranked_chunks, scores
    
    return specific_chunks[:top_k], [0.5] * min(top_k, len(specific_chunks))

def is_in_policy_content(query, chunks, scores, specific_policy=None):
    """Check if query matches policy content"""
    if specific_policy and chunks:
        return True
    
    query_normalized = query.lower().replace(' ', '').replace('-', '').replace('_', '')
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    
    if scores and scores[0] > 0.05:
        combined_text = " ".join(chunks[:2]).lower()
        matches = sum(1 for w in query_words if w in combined_text)
        if matches >= 1:
            return True
    
    all_policy_text = " ".join(st.session_state.policy_texts.values()).lower()
    keyword_matches = sum(1 for w in query_words if w in all_policy_text)
    
    if keyword_matches >= 2:
        return True
    
    return False

def get_suggestive_response(query, context, is_followup, client):
    """
    Generate a suggestive response based on policy content.
    Instead of generic welcome, analyze and suggest specific actions.
    """
    try:
        system_prompt = """You are a STRICT HR Policy Assistant for Spectron. 
Your job is to ANALYZE the user's question, SEARCH the policy context, and PROVIDE A SPECIFIC SUGGESTION.
NEVER give generic welcome messages. ALWAYS analyze the policy and suggest the specific action/leave type."""

        user_prompt = f"""Analyze this employee question and provide a SPECIFIC SUGGESTIVE response based ONLY on the policy context.

EMPLOYEE QUESTION: {query}

POLICY CONTEXT:
{context[:2500]}

INSTRUCTIONS:
1. First, analyze what the employee is asking for
2. Search the policy context for the specific answer
3. Provide a DIRECT SUGGESTION (e.g., "You may apply for Privilege Leave" or "You should submit a medical certificate for Sick Leave")
4. Include the PROCEDURE/STEPS from the policy
5. Mention any ELIGIBILITY CRITERIA or CONDITIONS
6. If multiple options exist, explain which is most suitable and why
7. If information is NOT in the policy, say: NOT_IN_POLICY
8. NEVER give generic welcomes like "I can help you with HR policies..."
9. ALWAYS be specific and actionable

EXAMPLE GOOD RESPONSES:
- "Based on our Privilege Leave Policy, you may apply for Privilege Leave for 2 weeks. Here is the procedure: [steps from policy]"
- "For your medical leave request, you should apply for Sick Leave. Requirements: [from policy]"
- "According to the Travel Policy, you can claim reimbursement by: [procedure]"

YOUR SUGGESTIVE RESPONSE:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=800
        )
        
        answer = response.choices[0].message.content.strip()
        
        if "NOT_IN_POLICY" in answer.upper() or len(answer) < 20:
            return None
        
        return answer
        
    except Exception as e:
        return None

# ============== UI COMPONENTS ==============
def show_logo():
    try:
        logo_path = Path("Logo.jpg")
        if logo_path.exists():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(str(logo_path), width=200)
        else:
            st.markdown("""
                <div style="text-align: center; margin-bottom: 0.5rem;">
                    <div style="font-size: 2rem; font-weight: 800; color: #c53030; letter-spacing: 3px;">SPECTRON</div>
                    <div style="font-size: 0.8rem; color: #718096;">HR POLICY ASSISTANT</div>
                </div>
            """, unsafe_allow_html=True)
    except:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 0.5rem;">
                <div style="font-size: 2rem; font-weight: 800; color: #c53030; letter-spacing: 3px;">SPECTRON</div>
            </div>
        """, unsafe_allow_html=True)

def show_welcome():
    """Show welcome screen - NO generic greeting, just intro"""
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to Your HR Assistant</div>
            <div class="welcome-text">
                Ask me about Spectron's HR policies. I'll analyze our policy documents and suggest the best course of action for you.
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_policy_links():
    """Show available policy documents with scrolling"""
    if not st.session_state.policy_files:
        return
    
    st.markdown('<div class="policy-links-section">', unsafe_allow_html=True)
    st.markdown('<div class="policy-links-title">📋 Available Policy Documents</div>', unsafe_allow_html=True)
    
    policy_names = []
    for pdf in st.session_state.policy_files:
        name = pdf.name.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
        policy_names.append(name)
    
    st.markdown('<div class="policy-list-container">', unsafe_allow_html=True)
    for name in sorted(policy_names):
        st.markdown(f'<div class="policy-item">📄 {name}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_chat():
    """Display chat messages - NO EMPTY STATE"""
    if not st.session_state.chat_history:
        return
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-header">👤 You</div>
                    {msg['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            content = msg['content']
            
            if content == "DECLINE":
                st.markdown("""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        <div class="decline-box">
                            I don't have specific information about this in our policy documents.
                            Please contact HR at <strong>hrd@spectron.in</strong> or <strong>+91 22 4606 6960 EXTN: 247</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Format policy answer with suggestion styling
                formatted = content.replace('\n', '<br>')
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        <div class="policy-answer">{formatted}</div>
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_typing():
    st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="color: #718096; font-size: 0.8rem;">Analyzing policies...</span>
        </div>
    """, unsafe_allow_html=True)

def show_contact():
    st.markdown("""
        <div class="contact-card">
            <div class="contact-title">📞 Contact HR</div>
            <div><strong>Email:</strong> hrd@spectron.in</div>
            <div><strong>Phone:</strong> +91 22 4606 6960 EXTN: 247</div>
            <div><strong>Hours:</strong> Mon - Sat, 10 AM to 6 PM</div>
        </div>
    """, unsafe_allow_html=True)

# ============== QUERY HANDLING ==============
def process_response(query):
    """Generate SUGGESTIVE response based on policy analysis"""
    
    # Check if greeting - but still analyze if it contains a question
    if is_greeting(query) and len(query.split()) <= 2:
        # Pure greeting - show welcome suggestion
        welcome_response = """Hello! 👋 

I'm your HR Policy Assistant. I can analyze Spectron's policy documents and suggest the best course of action for your HR queries.

**For example, you can ask:**
- "I want to take 2 weeks leave in May, which leave should I apply for?"
- "How do I claim travel reimbursement?"
- "What is the notice period policy?"

Please share your specific HR question and I'll search our policies to give you a tailored suggestion."""
        
        st.session_state.chat_history.append({"role": "assistant", "content": welcome_response})
        st.session_state.show_typing = False
        return
    
    # Try to find specific matching policy file
    specific_policy = find_matching_policy_file(query)
    
    chunks = []
    scores = []
    
    if specific_policy:
        chunks, scores = get_chunks_from_specific_policy(specific_policy, query)
        if not chunks:
            specific_policy = None
    
    if not specific_policy:
        chunks, scores = search_policy(query)
    
    # Check if actually in policy
    if not is_in_policy_content(query, chunks, scores, specific_policy):
        st.session_state.chat_history.append({"role": "assistant", "content": "DECLINE"})
        st.session_state.show_typing = False
        return
    
    context = "\n\n".join(chunks[:5])
    
    # Check if follow-up
    user_messages = [m for m in st.session_state.chat_history if m['role'] == 'user']
    is_followup = len(user_messages) > 1
    
    # Get suggestive response from OpenAI
    client = st.session_state.openai_client
    if client:
        response = get_suggestive_response(query, context, is_followup, client)
        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "DECLINE"})
    else:
        # No API - show raw context with suggestion format
        fallback = f"""Based on our policy documents, here is what I found:

{context[:600]}...

**Suggestion:** Please review the above policy information. For specific guidance, contact HR at hrd@spectron.in"""
        st.session_state.chat_history.append({"role": "assistant", "content": fallback})
    
    st.session_state.show_typing = False

# ============== MAIN ==============
def main():
    init_session_state()
    
    # Load policies (SILENT)
    if not st.session_state.policies_loaded:
        chunks, sources, policy_texts, pdf_files = load_policies()
        if chunks:
            st.session_state.policy_chunks = chunks
            st.session_state.policy_sources = sources
            st.session_state.policy_texts = policy_texts
            st.session_state.policy_files = pdf_files
            st.session_state.vectorizer, st.session_state.tfidf_matrix = setup_vectorizer(chunks)
            st.session_state.policies_loaded = True
    
    # Setup OpenAI
    if st.session_state.openai_client is None:
        st.session_state.openai_client = setup_openai()
    
    # Main content
    show_logo()
    st.markdown('<div class="sub-header">Your 24/7 AI-powered HR companion</div>', unsafe_allow_html=True)
    
    # Welcome (only if no chat)
    if not st.session_state.chat_history:
        show_welcome()
    
    # Policy links
    show_policy_links()
    
    # Chat display
    display_chat()
    
    if st.session_state.show_typing:
        show_typing()
    
    # CHECK FOR PENDING QUERY
    if 'pending_query' in st.session_state and st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None
        
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.show_typing = True
        
        process_response(query)
        st.rerun()
    
    # Input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    input_key = f"query_input_{st.session_state.input_counter}"
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        def on_input_change():
            val = st.session_state.get(input_key, "").strip()
            if val:
                st.session_state.pending_query = val
                st.session_state.input_counter += 1
        
        st.text_input(
            "Ask about HR policies...",
            key=input_key,
            placeholder="E.g., I want to take 2 weeks leave in May, which leave should I apply for?",
            label_visibility="collapsed",
            on_change=on_input_change
        )
    
    with col2:
        if st.button("🚀 Ask", use_container_width=True, type="primary"):
            val = st.session_state.get(input_key, "").strip()
            if val:
                st.session_state.pending_query = val
                st.session_state.input_counter += 1
                st.rerun()
    
    col3, col4 = st.columns([1, 1])
    with col3:
        if st.button("🔄 Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.input_counter += 1
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact card
    show_contact()
    
    # Footer
    st.markdown("""
        <div style="text-align: center; color: #a0aec0; font-size: 0.75rem; margin-top: 1rem;">
            🕐 Available 24/7 | 🔒 Private & Secure | © 2025 Spectron
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
