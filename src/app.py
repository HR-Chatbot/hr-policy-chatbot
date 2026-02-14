"""
HR Policy Chatbot for Spectron
Features: Professional UI, Mobile-friendly, Better UX, Error handling, Navigation
"""

import streamlit as st
import os
from pathlib import Path
from google import genai
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
        text-decoration: none;
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
        min-height: 300px;
        max-height: 500px;
        overflow-y: auto;
    }
    
    .empty-state {
        text-align: center;
        color: #a0aec0;
        padding: 3rem 1rem;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
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
        text-align: left;
        width: 100%;
        border: none;
        color: #2d3748;
        font-size: 0.95rem;
    }
    
    .faq-item:hover {
        background: #edf2f7;
        transform: translateX(5px);
    }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-top: 4px solid #c53030;
    }
    
    .stats-number {
        font-size: 2rem;
        font-weight: 700;
        color: #c53030;
    }
    
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.875rem;
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    .back-button {
        background: linear-gradient(135deg, #718096 0%, #4a5568 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
        text-decoration: none;
    }
    
    .back-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #4a5568 0%, #2d3748 100%);
    }
    
    .new-chat-button {
        background: linear-gradient(135deg, #c53030 0%, #9b2c2c 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
    }
    
    .new-chat-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(197, 48, 48, 0.3);
    }
    
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
    
    div[data-testid="stButton"] > button {
        width: 100%;
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
        'genai_client': None,
        'show_typing': False,
        'show_welcome': True,
        'pending_question': None
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

def chunk_text(text, chunk_size=800, overlap=150):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 100:
            chunks.append(chunk)
    return chunks

def load_policies():
    policies_dir = Path("policies")
    if not policies_dir.exists():
        st.error("❌ Policies folder not found!")
        return [], [], []
    
    pdf_files = list(policies_dir.glob("*.pdf"))
    if not pdf_files:
        st.warning("⚠️ No PDF files found")
        return [], [], []
    
    all_chunks, chunk_sources = [], []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pdf_file in enumerate(pdf_files):
        status_text.text(f"📄 Loading: {pdf_file.name}...")
        progress_bar.progress((idx + 1) / len(pdf_files))
        
        text = extract_text_from_pdf(pdf_file)
        if text:
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_sources.append(pdf_file.name)
    
    progress_bar.empty()
    status_text.empty()
    return all_chunks, chunk_sources, pdf_files

# ============== SEARCH FUNCTIONALITY ==============
def setup_vectorizer(chunks):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def find_relevant_chunks(query, top_k=3):
    if st.session_state.vectorizer is None or not st.session_state.policy_chunks:
        return [], []
    query_vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = [st.session_state.policy_chunks[i] for i in top_indices]
    scores = similarities[top_indices]
    return relevant_chunks, scores

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
            role = "👤 Employee" if msg['role'] == 'user' else "🤖 HR Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        prompt = f"""You are a professional HR Policy Assistant for Spectron company. Provide helpful, accurate responses based on company policies.

CONTEXT FROM COMPANY POLICY DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION: {query}

INSTRUCTIONS:
1. Answer based on the provided policy context first
2. If information is not in policies, provide general Indian HR best practices
3. Keep responses professional, concise, and friendly
4. For leave questions, mention applying through HR portal/manager
5. If unsure, say "Please contact HR directly for specific assistance"
6. Use bullet points for clarity when listing information

Provide a helpful response:"""
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt
        )
        return response.text
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return "API_QUOTA_EXHAUSTED"
        return f"Error processing request: {error_msg}"

# ============== CHAT PROCESSING ==============
def process_user_input(user_input):
    """Process user input and generate response"""
    if not user_input.strip():
        return
    
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.show_welcome = False
    
    # Show typing indicator
    st.session_state.show_typing = True
    
    # Get relevant context
    relevant_chunks, scores = find_relevant_chunks(user_input)
    context = "\n\n".join(relevant_chunks) if relevant_chunks else "No specific policy context found."
    
    # Generate response
    if st.session_state.genai_client:
        response = get_gemini_response(
            user_input, 
            context, 
            st.session_state.chat_history[:-1],
            st.session_state.genai_client
        )
    else:
        response = "⚠️ AI service not configured. Please contact HR directly at hrd@spectron.in"
    
    # Add bot response
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.show_typing = False

# ============== UI COMPONENTS ==============
def show_logo():
    """Display logo at the top"""
    try:
        logo_path = Path("Logo.jpg")
        if logo_path.exists():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(str(logo_path), width=250)
        else:
            show_text_logo()
    except:
        show_text_logo()

def show_text_logo():
    """Display text-based logo"""
    st.markdown("""
        <div style="text-align: center; margin-bottom: 1rem;">
            <div style="font-size: 2.5rem; font-weight: 800; color: #c53030; letter-spacing: 3px; text-transform: uppercase;">
                SPECTRON
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_back_button():
    """Display back button to return to welcome screen"""
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back", key="back_button", use_container_width=True):
            st.session_state.show_welcome = True
            st.session_state.chat_history = []
            st.session_state.pending_question = None
            st.rerun()

def show_new_chat_button():
    """Display new chat button in sidebar"""
    if st.button("🔄 New Chat", key="new_chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.show_welcome = True
        st.session_state.pending_question = None
        st.rerun()

def show_welcome_screen():
    """Display welcome screen with clickable example questions"""
    show_logo()
    
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to Your HR Assistant</div>
            <div class="welcome-text">
                I'm here to help you with HR policies, leave applications, benefits, and more. 
                Get instant answers to your HR questions, available 24/7 for your convenience!
            </div>
            <div class="example-questions">
                <div class="example-questions-title">💡 Try asking:</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Clickable example questions
    example_questions = [
        "How many casual leaves do I have per year?",
        "What is the notice period policy?",
        "How do I apply for medical leave?",
        "What are the company working hours?",
        "How to claim medical reimbursement?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(question, key=f"example_{i}", use_container_width=True):
            st.session_state.pending_question = question
            st.session_state.show_welcome = False
            st.rerun()

def display_chat_history():
    """Display chat messages"""
    if not st.session_state.chat_history:
        st.markdown("""
            <div class="chat-container">
                <div class="empty-state">
                    <div class="empty-state-icon">💬</div>
                    <div style="font-size: 1.2rem; font-weight: 500; color: #4a5568; margin-bottom: 0.5rem;">
                        No messages yet
                    </div>
                    <div style="color: #a0aec0;">
                        Start by asking a question about HR policies, leave, or benefits
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
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
            if message['content'] == "API_QUOTA_EXHAUSTED":
                st.markdown("""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        <div style="color: #c53030; font-weight: 500; margin-bottom: 0.5rem;">
                            ⚠️ Service temporarily unavailable due to high demand
                        </div>
                        <div style="font-size: 0.95rem;">
                            Please contact HR directly for immediate assistance:<br>
                            📧 <strong>hrd@spectron.in</strong><br>
                            📞 <strong>+91 22 4606 6960 EXTN: 247</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def show_typing_indicator():
    """Display typing animation"""
    st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="color: #718096; font-size: 0.9rem; margin-left: 0.5rem;">Thinking...</span>
        </div>
    """, unsafe_allow_html=True)

def show_contact_hr_card():
    """Display contact information card"""
    st.markdown("""
        <div class="contact-hr-card">
            <div class="contact-hr-title">📞 Need Personal Assistance?</div>
            <div style="font-size: 1rem; line-height: 1.8;">
                <div class="contact-detail">
                    <strong>HR Department - Spectron</strong>
                </div>
                <div class="contact-detail">
                    📧 <strong>hrd@spectron.in</strong>
                </div>
                <div class="contact-detail">
                    📞 <strong>+91 22 4606 6960 EXTN: 247</strong>
                </div>
                <div class="contact-detail">
                    🕐 <strong>Mon - Sat, 10 AM to 6 PM</strong>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_chat_interface():
    """Display chat interface with input and messages"""
    show_back_button()
    
    # Display chat history
    display_chat_history()
    
    # Show typing indicator if processing
    if st.session_state.show_typing:
        show_typing_indicator()
    
    # Chat input
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    user_input = st.chat_input("Type your HR question here...", key="chat_input")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if user_input:
        process_user_input(user_input)
        st.rerun()
    
    # Process pending question from example/FAQ click
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None
        process_user_input(question)
        st.rerun()
    
    show_contact_hr_card()

# ============== SIDEBAR ==============
def show_sidebar():
    """Display sidebar with stats, FAQs, and contact info"""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Company branding
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 2px solid #e2e8f0;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #c53030; letter-spacing: 2px;">
                    SPECTRON
                </div>
                <div style="font-size: 0.8rem; color: #718096;">
                    HR Assistant
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # New Chat Button
        show_new_chat_button()
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{len(st.session_state.policy_chunks)}</div>
                    <div style="font-size: 0.875rem; color: #718096;">Policy Sections</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{len([m for m in st.session_state.chat_history if m['role'] == 'user'])}</div>
                    <div style="font-size: 0.875rem; color: #718096;">Questions Asked</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Frequently Asked Questions (Clickable)
        st.markdown("### ❓ Frequently Asked")
        faqs = [
            "How do I apply for leave?",
            "What is the notice period?",
            "How many casual leaves per year?",
            "What are company working hours?",
            "How to claim medical reimbursement?"
        ]
        
        for i, faq in enumerate(faqs):
            if st.button(faq, key=f"faq_{i}", use_container_width=True):
                st.session_state.pending_question = faq
                st.session_state.show_welcome = False
                st.rerun()
        
        st.markdown("---")
        
        # Contact HR
        st.markdown("### 📞 Contact HR")
        st.markdown("""
            <div style="background: #f7fafc; padding: 1rem; border-radius: 12px; font-size: 0.9rem; line-height: 1.6;">
                <div style="margin-bottom: 0.5rem;"><strong>HR Department</strong></div>
                <div>📧 hrd@spectron.in</div>
                <div>📞 +91 22 4606 6960</div>
                <div>Ext: 247</div>
                <div style="margin-top: 0.5rem; color: #718096; font-size: 0.8rem;">
                    Mon - Sat<br>10 AM to 6 PM
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
            <div style="text-align: center; color: #a0aec0; font-size: 0.75rem; margin-top: 1rem;">
                © 2024 Spectron<br>All rights reserved
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============== MAIN APPLICATION ==============
def main():
    """Main application entry point"""
    init_session_state()
    
    # Load policies on first run
    if not st.session_state.policies_loaded:
        with st.spinner("Loading HR policies..."):
            chunks, sources, files = load_policies()
            st.session_state.policy_chunks = chunks
            st.session_state.policy_sources = sources
            st.session_state.vectorizer, st.session_state.tfidf_matrix = setup_vectorizer(chunks)
            st.session_state.genai_client = setup_gemini()
            st.session_state.policies_loaded = True
    
    # Show sidebar
    show_sidebar()
    
    # Main content area
    if st.session_state.show_welcome:
        show_welcome_screen()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()
