"""
HR Policy Chatbot for Spectron - Strict Policy-Only Version
Flow: Greeting → Check Policy → Answer/Decline → Follow-up with Policy Check
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
    
    .decline-box {
        background: #fff5f5;
        border-left: 3px solid #c53030;
        padding: 0.75rem;
        border-radius: 6px;
        color: #c53030;
        font-size: 0.9rem;
    }
    
    .greeting-box {
        background: #f0fff4;
        border-left: 3px solid #38a169;
        padding: 0.75rem;
        border-radius: 6px;
        color: #2f855a;
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
    .stDeployButton, #MainMenu, footer, header, .stSpinner, [data-testid="stSidebar"] {
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

def get_greeting_response():
    """Return greeting message"""
    return """Hello! 👋 Welcome to the Spectron HR Assistant.

I can help you with HR policies including:
• Leave policies (casual, sick, annual, maternity)
• Attendance and working hours
• Reimbursement procedures
• Notice period and resignation
• Code of conduct and discipline

Please ask me about any HR policy!"""

def is_in_policy_content(query, chunks, scores):
    """Check if query matches policy content OR filename - FIXED VERSION"""
    
    # Normalize query for matching
    query_normalized = query.lower().replace(' ', '').replace('-', '').replace('_', '')
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    
    # 1. Check if query matches any policy filename
    for pdf_file in st.session_state.policy_files:
        filename_normalized = pdf_file.name.lower().replace('.pdf', '').replace(' ', '').replace('-', '').replace('_', '')
        # Check if significant parts of query match filename
        if any(word in filename_normalized for word in query_words):
            return True
        # Check if filename contains query parts
        if len(query_normalized) > 5 and query_normalized in filename_normalized:
            return True
        # Check reverse: if filename keywords are in query
        filename_words = set(filename_normalized.split())
        if len(filename_words) > 0:
            match_count = sum(1 for fw in filename_words if fw in query_normalized and len(fw) > 3)
            if match_count >= 1:
                return True
    
    # 2. Check semantic similarity in content
    if scores and scores[0] > 0.05:  # Lowered threshold
        combined_text = " ".join(chunks[:2]).lower()
        matches = sum(1 for w in query_words if w in combined_text)
        if matches >= 1:
            return True
    
    # 3. Check if query keywords exist in any policy text
    all_policy_text = " ".join(st.session_state.policy_texts.values()).lower()
    keyword_matches = sum(1 for w in query_words if w in all_policy_text)
    if keyword_matches >= 2:  # At least 2 keywords found
        return True
    
    return False

def format_policy_response(query, context, is_followup, client):
    """Use OpenAI to format policy content - STRICTLY no external knowledge"""
    try:
        system_prompt = """You are a STRICT HR Policy Assistant. You must ONLY use the provided policy context.
NEVER use external knowledge. If answer not in context, say "NOT_IN_POLICY"."""

        if is_followup:
            user_prompt = f"""This is a FOLLOW-UP question. Answer ONLY using the policy context.

Policy Context:
{context[:1500]}

Follow-up Question: {query}

Instructions:
1. Answer ONLY if information exists in the context above
2. If not found, respond with exactly: NOT_IN_POLICY
3. Be concise (2-3 sentences max)
4. No external knowledge allowed

Answer:"""
        else:
            user_prompt = f"""Format this policy information clearly.

Policy Context:
{context[:2000]}

Question: {query}

Instructions:
1. Create a structured response with these sections if available:
   - **Purpose**: Why this policy exists
   - **Eligibility**: Who is covered
   - **Entitlement**: What is provided (days, amounts, etc.)
   - **Application**: How to apply/request
   - **Approval**: Who approves
   - **Important Notes**: Key restrictions or rules
2. Use bullet points
3. If information is missing for a section, skip it
4. If NO relevant information, say: NOT_IN_POLICY

Answer:"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,  # Zero temperature for strict adherence
            max_tokens=500
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
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to Your HR Assistant</div>
            <div class="welcome-text">
                Get instant answers about HR policies, leaves, benefits, and more. Available 24/7!
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_policy_links():
    """Show available policy documents with scrolling (replacing FAQ)"""
    if not st.session_state.policy_files:
        return
    
    st.markdown('<div class="policy-links-section">', unsafe_allow_html=True)
    st.markdown('<div class="policy-links-title">📋 Available Policy Documents</div>', unsafe_allow_html=True)
    
    # Format policy filenames nicely
    policy_names = []
    for pdf in st.session_state.policy_files:
        # Remove .pdf and replace underscores/hyphens with spaces
        name = pdf.name.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
        policy_names.append(name)
    
    # Display as scrollable list
    st.markdown('<div class="policy-list-container">', unsafe_allow_html=True)
    for name in sorted(policy_names):
        st.markdown(f'<div class="policy-item">📄 {name}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def display_chat():
    """Display chat messages - NO EMPTY STATE"""
    if not st.session_state.chat_history:
        return  # Don't show anything if no messages
    
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
                            I apologize, but I don't have information about this in our policy documents. 
                            Please contact HR at <strong>hrd@spectron.in</strong> or <strong>+91 22 4606 6960 EXTN: 247</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            elif content == "GREETING":
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        <div class="greeting-box">{get_greeting_response()}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Format policy answer
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
            <span style="color: #718096; font-size: 0.8rem;">Searching policy...</span>
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
    """Generate response based on flowchart logic"""
    
    # 1. Check if greeting
    if is_greeting(query):
        st.session_state.chat_history.append({"role": "assistant", "content": "GREETING"})
        st.session_state.show_typing = False
        return
    
    # 2. Search policy
    chunks, scores = search_policy(query)
    
    # 3. Check if actually in policy (strict but includes filename matching)
    if not is_in_policy_content(query, chunks, scores):
        st.session_state.chat_history.append({"role": "assistant", "content": "DECLINE"})
        st.session_state.show_typing = False
        return
    
    # 4. Build context from relevant chunks
    context = "\n\n".join(chunks[:3])
    
    # 5. Check if follow-up
    user_messages = [m for m in st.session_state.chat_history if m['role'] == 'user']
    is_followup = len(user_messages) > 1
    
    # 6. Get formatted response from OpenAI
    client = st.session_state.openai_client
    if client:
        response = format_policy_response(query, context, is_followup, client)
        
        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "DECLINE"})
    else:
        # No API - show raw context
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"Based on policy:\n\n{context[:600]}..."
        })
    
    st.session_state.show_typing = False

# ============== MAIN ==============

def main():
    init_session_state()
    
    # Load policies (SILENT - no message)
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
    
    # Policy links (replacing FAQ) - NOW WITH SCROLLING
    show_policy_links()
    
    # Chat display (NO EMPTY STATE)
    display_chat()
    
    if st.session_state.show_typing:
        show_typing()
    
    # CHECK FOR PENDING QUERY (from Enter key or button)
    if 'pending_query' in st.session_state and st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None  # Clear it
        
        # Add user message and show typing
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.show_typing = True
        
        # Process response
        process_response(query)
        
        # Rerun to update UI
        st.rerun()
    
    # Input area - AUTO SUBMIT ON ENTER + AUTO CLEAR
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    # Dynamic key for auto-clear
    input_key = f"query_input_{st.session_state.input_counter}"
    
    # Use columns for layout
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Text input - on_change sets pending_query, NO rerun in callback
        def on_input_change():
            val = st.session_state.get(input_key, "").strip()
            if val:
                st.session_state.pending_query = val
                st.session_state.input_counter += 1  # Clear input next run
        
        st.text_input(
            "Ask about HR policies...",
            key=input_key,
            placeholder="E.g., How many casual leaves do I have?",
            label_visibility="collapsed",
            on_change=on_input_change
        )
    
    with col2:
        # Ask button - also sets pending_query
        if st.button("🚀 Ask", use_container_width=True, type="primary"):
            val = st.session_state.get(input_key, "").strip()
            if val:
                st.session_state.pending_query = val
                st.session_state.input_counter += 1
                st.rerun()  # Rerun immediately for button
    
    # Action buttons
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
