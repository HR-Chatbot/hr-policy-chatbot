"""
HR Policy Chatbot for Spectron - Strict Policy-Only Version
Features: Policy-only answers, conversational greetings, auto-submit, clean UI
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
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a365d;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .logo-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .welcome-text {
        text-align: center;
        font-size: 1.05rem;
        line-height: 1.6;
        opacity: 0.95;
    }
    
    .example-questions {
        background: rgba(255,255,255,0.15);
        padding: 1.25rem;
        border-radius: 12px;
        margin-top: 1.5rem;
    }
    
    .example-questions-title {
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    
    .example-question {
        display: block;
        background: rgba(255,255,255,0.95);
        color: #2d3748;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        border: none;
        width: 100%;
    }
    
    .example-question:hover {
        background: white;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .chat-container {
        background: #f7fafc;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        min-height: 200px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        max-width: 85%;
        animation: fadeIn 0.3s ease;
        line-height: 1.5;
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
        box-shadow: 0 4px 15px rgba(197, 48, 48, 0.3);
    }
    
    .bot-message {
        background: white;
        color: #2d3748;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }
    
    .message-header {
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }
    
    .input-container {
        background: white;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #c53030;
        box-shadow: 0 0 0 3px rgba(197, 48, 48, 0.1);
    }
    
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem 1.25rem;
        background: white;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        width: fit-content;
        margin-bottom: 1rem;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #c53030;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-10px); }
    }
    
    .policy-answer {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        line-height: 1.8;
    }
    
    .policy-answer h4 {
        color: #1a365d;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .policy-answer ul {
        margin-left: 1.5rem;
    }
    
    .policy-answer li {
        margin-bottom: 0.5rem;
    }
    
    .decline-message {
        background: #fff5f5;
        border-left: 4px solid #c53030;
        padding: 1rem;
        border-radius: 8px;
        color: #c53030;
    }
    
    .greeting-message {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 1rem;
        border-radius: 8px;
        color: #2f855a;
    }
    
    .contact-hr-card {
        background: linear-gradient(135deg, #f6e05e 0%, #d69e2e 100%);
        color: #744210;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
    }
    
    .contact-hr-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    
    .contact-detail {
        margin: 0.5rem 0;
        font-size: 1rem;
    }
    
    .sidebar-content {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }
    
    .faq-item {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border-left: 4px solid #c53030;
    }
    
    .faq-item:hover {
        background: #edf2f7;
        transform: translateX(5px);
    }
    
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.875rem;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Remove extra spacing */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Hide Streamlit default elements */
    .stDeployButton {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.75rem;
        }
        .chat-message {
            max-width: 95%;
            font-size: 0.95rem;
        }
        .welcome-card {
            padding: 1.5rem;
        }
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
        'show_welcome': True,
        'policy_texts': {},
        'suggested_questions': [],
        'input_key': 0  # For clearing input
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============== PDF PROCESSING ==============

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
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
        return [], [], [], {}
    
    pdf_files = list(policies_dir.glob("*.pdf"))
    if not pdf_files:
        return [], [], [], {}
    
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
    
    return all_chunks, chunk_sources, pdf_files, policy_texts

# ============== SEARCH FUNCTIONALITY ==============

def setup_vectorizer(chunks):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def find_relevant_chunks(query, top_k=5):
    if st.session_state.vectorizer is None or not st.session_state.policy_chunks:
        return [], []
    query_vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [st.session_state.policy_chunks[i] for i in top_indices]
    scores = similarities[top_indices]
    return relevant_chunks, scores

# ============== QUERY CLASSIFICATION ==============

def is_greeting(query):
    """Check if query is a greeting"""
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'namaste', 'greetings']
    query_lower = query.lower().strip()
    return any(g in query_lower for g in greetings) or len(query_lower) < 4

def is_policy_related(query):
    """Check if query is related to HR policies"""
    policy_keywords = [
        'leave', 'attendance', 'salary', 'pay', 'reimbursement', 'claim', 'benefit', 'policy',
        'holiday', 'vacation', 'sick', 'medical', 'casual', 'annual', 'privilege', 'maternity',
        'paternity', 'notice', 'resignation', 'termination', 'probation', 'confirmation',
        'working hours', 'overtime', 'shift', 'allowance', 'deduction', 'tax', 'insurance',
        'pf', 'provident fund', 'gratuity', 'bonus', 'increment', 'appraisal', 'promotion',
        'transfer', 'training', 'travel', 'expense', 'code of conduct', 'disciplinary',
        'harassment', 'grievance', 'complaint', 'safety', 'security', 'asset', 'laptop'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in policy_keywords)

def check_policy_content(query, threshold=0.1):
    """Check if query matches actual policy content"""
    relevant_chunks, scores = find_relevant_chunks(query, top_k=3)
    best_score = scores[0] if len(scores) > 0 else 0
    
    # Also check for keyword presence
    query_words = set(query.lower().split())
    policy_text = " ".join(st.session_state.policy_texts.values()).lower()
    word_matches = sum(1 for word in query_words if len(word) > 3 and word in policy_text)
    
    has_match = best_score > threshold or word_matches >= 2
    return has_match, relevant_chunks, scores

# ============== EXTRACT FAQ FROM POLICIES ==============

def extract_suggested_questions():
    """Extract potential questions from policy content"""
    questions = []
    
    patterns = [
        (r'(?i)(casual leave)', 'How many casual leaves do I have per year?'),
        (r'(?i)(sick leave|medical leave)', 'How many sick leaves do I have?'),
        (r'(?i)(annual leave|privilege leave)', 'What is the annual leave entitlement?'),
        (r'(?i)(notice period|resignation)', 'What is the notice period policy?'),
        (r'(?i)(apply for leave|leave application)', 'How do I apply for leave?'),
        (r'(?i)(working hours|office hours)', 'What are the company working hours?'),
        (r'(?i)(reimbursement|medical claim)', 'How to claim medical reimbursement?'),
        (r'(?i)(probation|confirmation)', 'What is the probation period?'),
        (r'(?i)(attendance|biometric)', 'What is the attendance policy?'),
    ]
    
    all_text = " ".join(st.session_state.policy_texts.values()).lower()
    
    for pattern, question in patterns:
        if re.search(pattern, all_text) and question not in questions:
            questions.append(question)
    
    # Add defaults if needed
    defaults = [
        "How many casual leaves do I have per year?",
        "What is the notice period policy?",
        "How do I apply for leave?",
        "What are the company working hours?",
        "How to claim medical reimbursement?"
    ]
    
    for d in defaults:
        if d not in questions:
            questions.append(d)
    
    return questions[:5]

# ============== OPENAI SETUP ==============

def setup_openai():
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        return None

def get_greeting_response():
    """Return a friendly greeting"""
    return """Hello! 👋 Welcome to the Spectron HR Assistant.

I'm here to help you with HR policies, leave applications, benefits, and other HR-related queries. 

Please feel free to ask me about:
• Leave policies (casual, sick, annual)
• Attendance and working hours
• Reimbursement procedures
• Notice period and resignation
• Any other HR policy questions

How can I assist you today?"""

def format_policy_answer(query, context, client):
    """Format policy content into structured answer"""
    try:
        prompt = f"""You are an HR Policy Assistant. Create a clear, structured answer using ONLY the provided policy context.

Query: {query}

Policy Context:
{context[:2000]}

Instructions:
1. Use ONLY information from the context above
2. Format with clear headers like: **Purpose**, **Eligibility**, **Entitlement**, **Application**, **Approval**
3. Use bullet points for lists
4. If specific information is not in context, say "NOT_FOUND"
5. Be concise and professional

Answer:"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600
        )
        
        answer = response.choices[0].message.content.strip()
        
        if "NOT_FOUND" in answer or len(answer) < 30:
            return None
        
        return answer
        
    except Exception as e:
        return None

def get_policy_response(query, context, chat_history, client):
    """Get response for policy-related query"""
    try:
        # Build conversation context
        history_text = ""
        for msg in chat_history[-3:]:
            role = "User" if msg['role'] == 'user' else "Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        prompt = f"""You are a strict HR Policy Assistant. Answer using ONLY the policy context provided.

Conversation History:
{history_text}

Policy Context:
{context[:2000]}

Current Question: {query}

Instructions:
1. Answer ONLY based on the policy context above
2. If the answer is not in the policy, respond with "NOT_IN_POLICY"
3. Do not use external knowledge
4. Be concise and professional

Answer:"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400
        )
        
        answer = response.choices[0].message.content.strip()
        
        if "NOT_IN_POLICY" in answer:
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
                st.image(str(logo_path), width=250)
        else:
            st.markdown("""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <div style="font-size: 2.5rem; font-weight: 800; color: #c53030; letter-spacing: 3px; text-transform: uppercase;">
                        SPECTRON
                    </div>
                </div>
            """, unsafe_allow_html=True)
    except:
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1rem;">
                <div style="font-size: 2.5rem; font-weight: 800; color: #c53030; letter-spacing: 3px; text-transform: uppercase;">
                    SPECTRON
                </div>
            </div>
        """, unsafe_allow_html=True)

def show_welcome_screen():
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to Your HR Assistant</div>
            <div class="welcome-text">
                I'm here to help you with HR policies, leave applications, benefits, and more. 
                Get instant answers to your HR questions, available 24/7 for your convenience!
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_chat_history():
    if not st.session_state.chat_history:
        return
    
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="chat-message user-message">
                    <div class="message-header">👤 You</div>
                    {message['content']}
                </div>
            """, unsafe_allow_html=True)
        else:
            content = message['content']
            if content == "POLICY_NOT_FOUND":
                st.markdown("""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        <div class="decline-message">
                            <strong>I apologize,</strong> but I don't have information about this in our current policy documents. 
                            Please contact HR directly for assistance with this query.
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            elif content == "GREETING":
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        <div class="greeting-message">{get_greeting_response()}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                formatted = content.replace('\n', '<br>')
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        <div class="policy-answer">{formatted}</div>
                    </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_typing_indicator():
    st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="color: #718096; font-size: 0.9rem; margin-left: 0.5rem;">Thinking...</span>
        </div>
    """, unsafe_allow_html=True)

def show_contact_hr_card():
    st.markdown("""
        <div class="contact-hr-card">
            <div class="contact-hr-title">📞 Need Personal Assistance?</div>
            <div style="font-size: 1rem; line-height: 1.8;">
                <div class="contact-detail"><strong>HR Department - Spectron</strong></div>
                <div class="contact-detail">📧 <strong>hrd@spectron.in</strong></div>
                <div class="contact-detail">📞 <strong>+91 22 4606 6960 EXTN: 247</strong></div>
                <div class="contact-detail">🕐 <strong>Mon - Sat, 10 AM to 6 PM</strong></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 2px solid #e2e8f0;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #c53030; letter-spacing: 2px;">SPECTRON</div>
                <div style="font-size: 0.8rem; color: #718096;">HR Assistant</div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ❓ Frequently Asked")
        for faq in st.session_state.suggested_questions[:5]:
            st.markdown(f'<div class="faq-item">{faq}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### 📞 Contact HR")
        st.markdown("""
            <div style="background: #f7fafc; padding: 1rem; border-radius: 12px; font-size: 0.9rem;">
                <strong style="color: #c53030;">Spectron HR</strong><br>
                📧 hrd@spectron.in<br>
                📞 +91 22 4606 6960<br>
                &nbsp;&nbsp;&nbsp;&nbsp;EXTN: 247<br>
                🕐 Mon-Sat: 10 AM - 6 PM
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============== MAIN APP ==============

def main():
    init_session_state()
    
    # Header
    show_logo()
    st.markdown('<div class="sub-header">Your 24/7 AI-powered HR companion</div>', unsafe_allow_html=True)
    
    # Setup OpenAI
    if st.session_state.openai_client is None:
        st.session_state.openai_client = setup_openai()
    
    # Load policies (silent)
    if not st.session_state.policies_loaded:
        with st.spinner(""):
            chunks, sources, pdf_files, policy_texts = load_policies()
            if chunks:
                st.session_state.policy_chunks = chunks
                st.session_state.policy_sources = sources
                st.session_state.policy_texts = policy_texts
                st.session_state.vectorizer, st.session_state.tfidf_matrix = setup_vectorizer(chunks)
                st.session_state.policies_loaded = True
                st.session_state.suggested_questions = extract_suggested_questions()
    
    # Sidebar
    show_sidebar()
    
    # Welcome screen
    if st.session_state.show_welcome or not st.session_state.chat_history:
        show_welcome_screen()
        st.session_state.show_welcome = False
    
    # FAQ buttons
    if st.session_state.suggested_questions:
        st.markdown('<div class="example-questions">', unsafe_allow_html=True)
        st.markdown('<div class="example-questions-title">💡 Try asking:</div>', unsafe_allow_html=True)
        
        cols = st.columns(len(st.session_state.suggested_questions[:3]))
        for i, question in enumerate(st.session_state.suggested_questions[:3]):
            with cols[i]:
                if st.button(question, key=f"faq_btn_{i}", use_container_width=True):
                    process_message(question)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat display
    display_chat_history()
    
    if st.session_state.show_typing:
        show_typing_indicator()
    
    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Use dynamic key to clear input
    current_key = f"user_input_{st.session_state.input_key}"
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        query = st.text_input(
            "Ask your question...",
            placeholder="E.g., How many casual leaves do I have?",
            key=current_key,
            label_visibility="collapsed",
            on_change=lambda: handle_input_change(current_key)
        )
    
    with col2:
        submit = st.button("🚀 Ask", use_container_width=True, type="primary", 
                          on_click=lambda: handle_submit(current_key))
    
    col3, col4 = st.columns([1, 1])
    with col3:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.show_welcome = True
            st.rerun()
    with col4:
        if st.button("📞 Contact HR", use_container_width=True):
            pass
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process pending message
    if 'pending_message' in st.session_state and st.session_state.pending_message:
        msg = st.session_state.pending_message
        st.session_state.pending_message = None
        process_response(msg)
    
    # Contact card
    show_contact_hr_card()
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>🕐 Available 24/7 | 🔒 Conversations are private and secure</p>
            <p>⚠️ For complex issues, please contact HR directly at hrd@spectron.in</p>
            <p style="font-size: 0.75rem; color: #a0aec0; margin-top: 1rem;">© 2025 Spectron. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

def handle_input_change(key):
    """Handle Enter key press"""
    value = st.session_state.get(key, "").strip()
    if value:
        st.session_state.pending_message = value
        st.session_state.input_key += 1  # Clear input

def handle_submit(key):
    """Handle Ask button click"""
    value = st.session_state.get(key, "").strip()
    if value:
        st.session_state.pending_message = value
        st.session_state.input_key += 1  # Clear input

def process_message(message):
    """Process message from FAQ button"""
    st.session_state.pending_message = message
    st.session_state.input_key += 1
    st.rerun()

def process_response(query):
    """Process the query and generate response"""
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": query})
    st.session_state.show_typing = True
    
    # Check query type
    if is_greeting(query):
        # Greeting - respond friendly
        st.session_state.chat_history.append({"role": "assistant", "content": "GREETING"})
        st.session_state.show_typing = False
        st.rerun()
        return
    
    if not is_policy_related(query):
        # Not policy related - decline
        st.session_state.chat_history.append({"role": "assistant", "content": "POLICY_NOT_FOUND"})
        st.session_state.show_typing = False
        st.rerun()
        return
    
    # Check if content exists in policies
    has_match, relevant_chunks, scores = check_policy_content(query)
    
    if not has_match:
        # No policy match - decline
        st.session_state.chat_history.append({"role": "assistant", "content": "POLICY_NOT_FOUND"})
        st.session_state.show_typing = False
        st.rerun()
        return
    
    # Build context
    context = "\n\n".join([chunk for chunk in relevant_chunks[:3]])
    
    # Get AI formatted response
    client = st.session_state.openai_client
    if client:
        # Check if follow-up
        is_followup = len([m for m in st.session_state.chat_history if m['role'] == 'user']) > 1
        
        if is_followup:
            response = get_policy_response(query, context, st.session_state.chat_history[:-1], client)
        else:
            response = format_policy_answer(query, context, client)
        
        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            st.session_state.chat_history.append({"role": "assistant", "content": "POLICY_NOT_FOUND"})
    else:
        # No API - show raw context
        st.session_state.chat_history.append({"role": "assistant", "content": f"Based on our policies:\n\n{context[:800]}"})
    
    st.session_state.show_typing = False
    st.rerun()

if __name__ == "__main__":
    main()
