"""
HR Policy Chatbot for Spectron
Features: Policy-based answers only, Auto-scan all policies, Professional UI
"""

import streamlit as st
import os
from pathlib import Path

# Try different import methods for Google AI
try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    try:
        import google.generativeai as genai
        GENAI_AVAILABLE = True
    except ImportError:
        GENAI_AVAILABLE = False
        genai = None

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
    
    .company-name {
        font-size: 1.5rem;
        color: #c53030;
        text-align: center;
        font-weight: 600;
        margin-bottom: 0.5rem;
        letter-spacing: 2px;
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
    
    .policy-reference {
        background: #edf2f7;
        border-left: 4px solid #c53030;
        padding: 0.75rem 1rem;
        margin-top: 0.75rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: #4a5568;
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
        margin-bottom: 0.5rem;
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
    
    .no-policy-warning {
        background: #fed7d7;
        border: 1px solid #fc8181;
        color: #c53030;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
    }
    
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.75rem;
        }
        .company-name {
            font-size: 1.25rem;
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
        'genai_client': None,
        'show_typing': False,
        'show_welcome': True,
        'policy_names': [],
        'api_available': GENAI_AVAILABLE
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============== PDF PROCESSING ==============
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with error handling"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into meaningful chunks with overlap"""
    if not text:
        return []
    
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(para) > chunk_size:
            words = para.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 < chunk_size:
                    temp_chunk += word + " "
                else:
                    if temp_chunk.strip():
                        chunks.append(temp_chunk.strip())
                    temp_chunk = word + " "
            if temp_chunk.strip():
                current_chunk += temp_chunk.strip() + "\n\n"
        else:
            if len(current_chunk) + len(para) < chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def load_policies():
    """Load ALL PDF policies from the policies folder - auto-detects new files"""
    policies_dir = Path("policies")
    
    if not policies_dir.exists():
        st.error("❌ Policies folder not found!")
        return [], [], [], []
    
    pdf_files = sorted(list(policies_dir.glob("*.pdf")))
    
    if not pdf_files:
        st.warning("⚠️ No PDF policy files found")
        return [], [], [], []
    
    all_chunks = []
    chunk_sources = []
    policy_names = []
    
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            policy_names.append(pdf_file.name)
            chunks = chunk_text(text)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_sources.append(pdf_file.name)
    
    return all_chunks, chunk_sources, policy_names, pdf_files

# ============== SEARCH FUNCTIONALITY ==============
def setup_vectorizer(chunks):
    """Setup TF-IDF vectorizer for semantic search"""
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def find_relevant_chunks(query, top_k=5):
    """Find most relevant policy chunks for a query"""
    if st.session_state.vectorizer is None or not st.session_state.policy_chunks:
        return [], [], []
    
    query_vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
    
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    relevant_chunks = [st.session_state.policy_chunks[i] for i in top_indices]
    sources = [st.session_state.policy_sources[i] for i in top_indices]
    scores = [similarities[i] for i in top_indices]
    
    filtered = [(chunk, src, score) for chunk, src, score in zip(relevant_chunks, sources, scores) if score > 0.05]
    
    if not filtered:
        return [], [], []
    
    return [f[0] for f in filtered], [f[1] for f in filtered], [f[2] for f in filtered]
    # ============== GEMINI AI SETUP ==============
def setup_gemini():
    """Setup Gemini API client with fallback"""
    if not GENAI_AVAILABLE:
        return None
    
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except:
        api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        return None
    
    try:
        # Try new API first
        if hasattr(genai, 'Client'):
            client = genai.Client(api_key=api_key)
            return {'type': 'new', 'client': client}
        else:
            # Fall back to old API
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.5-pro')
            return {'type': 'old', 'model': model}
    except Exception as e:
        return None

def get_gemini_response(query, context, sources, chat_history, client_config):
    """
    Get response from Gemini based ONLY on policy documents.
    """
    if not client_config:
        return {
            "answer": "HR Assistant is currently unavailable. Please contact HR directly at hrd@spectron.in or call +91 22 4606 6960 EXTN: 247.",
            "sources": [],
            "has_context": False
        }
    
    try:
        # Build conversation history
        history_text = ""
        for msg in chat_history[-6:]:
            if msg['role'] == 'user':
                history_text += f"Employee: {msg['content']}\n"
            else:
                history_text += f"HR Assistant: {msg['content']}\n"
        
        unique_sources = list(dict.fromkeys(sources))
        has_relevant_context = len(context.strip()) > 50 and len(unique_sources) > 0
        
        if not has_relevant_context:
            return {
                "answer": "I don't have specific information about this in our policy documents. Please contact HR directly for assistance.",
                "sources": [],
                "has_context": False
            }
        
        sources_list = "\n".join([f"- {src}" for src in unique_sources[:3]])
        
        prompt = f"""You are the HR Policy Assistant for Spectron. You MUST answer based ONLY on the provided policy documents.

IMPORTANT RULES:
1. Answer ONLY using the information in the "CONTEXT FROM POLICY DOCUMENTS" below
2. If the answer is not in the context, say "I don't have specific information about this in our policy documents. Please contact HR directly for assistance."
3. Do NOT make up information or use general knowledge
4. Suggest the specific leave type or policy action based on the documents
5. Include the procedure/steps if mentioned in the policy
6. Keep responses professional and concise

AVAILABLE POLICY DOCUMENTS:
{sources_list}

CONTEXT FROM POLICY DOCUMENTS:
{context}

CONVERSATION HISTORY:
{history_text}

EMPLOYEE QUESTION: {query}

INSTRUCTIONS:
- Analyze the question carefully
- Check which policy document applies
- Provide specific suggestion (e.g., "You may apply for Privilege Leave")
- Include procedure from the policy if available
- If multiple options exist, explain the criteria for choosing

YOUR RESPONSE:"""

        # Use appropriate API
        if client_config['type'] == 'new':
            response = client_config['client'].models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt
            )
            answer = response.text
        else:
            response = client_config['model'].generate_content(prompt)
            answer = response.text
        
        return {
            "answer": answer,
            "sources": unique_sources[:3],
            "has_context": True
        }
        
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return {
                "answer": "API_QUOTA_EXHAUSTED",
                "sources": [],
                "has_context": False
            }
        return {
            "answer": f"I apologize, but I'm having trouble processing your request. Please contact HR directly at hrd@spectron.in or call +91 22 4606 6960 EXTN: 247.",
            "sources": [],
            "has_context": False
        }

def process_query(query, client_config):
    """Process user query and generate policy-based response"""
    relevant_chunks, sources, scores = find_relevant_chunks(query)
    
    context = "\n\n---\n\n".join([
        f"[From: {src}]\n{chunk[:1500]}"
        for chunk, src, score in zip(relevant_chunks, sources, scores)
    ])
    
    result = get_gemini_response(query, context, sources, st.session_state.chat_history, client_config)
    
    return result
    # ============== UI COMPONENTS ==============
def show_logo():
    """Display company logo"""
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
    """Show welcome screen with example questions"""
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to Your HR Assistant</div>
            <div class="welcome-text">
                Ask me about Spectron's HR policies. I'll search through our policy documents 
                and provide you with specific answers and suggestions.
            </div>
            <div class="example-questions">
                <div class="example-questions-title">💡 Try asking:</div>
                <div class="example-question">I want to take 2 weeks leave in May, which leave should I apply for?</div>
                <div class="example-question">How many casual leaves can I take per year?</div>
                <div class="example-question">What is the procedure to apply for sick leave?</div>
                <div class="example-question">How do I claim travel reimbursement?</div>
                <div class="example-question">What is the notice period policy?</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

def display_chat_history():
    """Display chat messages"""
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
            
            if content == "API_QUOTA_EXHAUSTED":
                st.markdown("""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        <div style="color: #c53030; font-weight: 500; margin-bottom: 0.5rem;">
                            ⚠️ Service temporarily unavailable
                        </div>
                        <div>
                            Please contact HR directly:<br>
                            📧 <strong>hrd@spectron.in</strong><br>
                            📞 <strong>+91 22 4606 6960 EXTN: 247</strong>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                msg_parts = content.split("||SOURCE||")
                answer = msg_parts[0]
                source_html = ""
                
                if len(msg_parts) > 1 and msg_parts[1]:
                    sources = msg_parts[1].split(",")
                    source_list = " • ".join([f"📄 {s.replace('.pdf', '')}" for s in sources[:2]])
                    source_html = f'<div class="policy-reference">Based on: {source_list}</div>'
                
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        {answer}
                        {source_html}
                    </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_typing_indicator():
    """Show typing animation"""
    st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="color: #718096; font-size: 0.9rem; margin-left: 0.5rem;">Searching policies...</span>
        </div>
    """, unsafe_allow_html=True)

def show_contact_hr_card():
    """Show contact HR information"""
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

# ============== SIDEBAR ==============
def show_sidebar():
    """Display sidebar with stats and FAQs"""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
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
        
        st.markdown("### 📊 Policy Database")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{len(st.session_state.policy_names)}</div>
                    <div style="font-size: 0.875rem; color: #718096;">Policies</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="stats-card">
                    <div class="stats-number">{len([m for m in st.session_state.chat_history if m['role'] == 'user'])}</div>
                    <div style="font-size: 0.875rem; color: #718096;">Questions</div>
                </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.policy_names:
            st.markdown("---")
            st.markdown("### 📋 Available Policies")
            for policy in st.session_state.policy_names[:5]:
                clean_name = policy.replace('.pdf', '').replace('_', ' ')
                st.markdown(f'<div style="font-size: 0.85rem; color: #4a5568; margin: 0.25rem 0;">• {clean_name}</div>', unsafe_allow_html=True)
            if len(st.session_state.policy_names) > 5:
                st.markdown(f'<div style="font-size: 0.8rem; color: #718096; margin-top: 0.5rem;">+{len(st.session_state.policy_names) - 5} more...</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("### ❓ Quick Questions")
        faqs = [
            "How do I apply for leave?",
            "What is the notice period?",
            "How many casual leaves per year?",
            "What are company working hours?",
            "How to claim reimbursement?"
        ]
        for faq in faqs:
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
    """Main application"""
    init_session_state()
    
    show_logo()
    st.markdown('<div class="sub-header">Your 24/7 Policy-Based HR Assistant</div>', unsafe_allow_html=True)
    
    if st.session_state.genai_client is None:
        st.session_state.genai_client = setup_gemini()
    
    if not st.session_state.policies_loaded:
        chunks, sources, policy_names, pdf_files = load_policies()
        if chunks:
            st.session_state.policy_chunks = chunks
            st.session_state.policy_sources = sources
            st.session_state.policy_names = policy_names
            st.session_state.vectorizer, st.session_state.tfidf_matrix = setup_vectorizer(chunks)
            st.session_state.policies_loaded = True
        else:
            st.error("❌ No policies found. Please upload PDF files to the policies folder.")
            return
    
    show_sidebar()
    
    if not st.session_state.chat_history:
        show_welcome_screen()
    
    display_chat_history()
    
    if st.session_state.show_typing:
        show_typing_indicator()
    
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        
        with col1:
            query = st.text_input(
                "Ask your question...",
                placeholder="E.g., I want to take 2 weeks leave in May, which leave should I apply for?",
                key="user_input",
                label_visibility="collapsed"
            )
        
        with col2:
            submit = st.form_submit_button(
                "🚀 Ask",
                use_container_width=True,
                type="primary"
            )
    
    col3, col4, col5 = st.columns([1, 1, 3])
    with col3:
        if st.button("🔄 Clear Chat", use_container_width=True, key="clear_btn"):
            st.session_state.chat_history = []
            st.session_state.show_welcome = True
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit and query and query.strip():
        st.session_state.chat_history.append({"role": "user", "content": query.strip()})
        st.session_state.show_typing = True
        st.rerun()
    
    if st.session_state.show_typing and st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]
        if last_msg['role'] == 'user':
            client_config = st.session_state.genai_client
            result = process_query(last_msg['content'], client_config)
            
            if result['has_context'] and result['sources']:
                sources_str = ",".join(result['sources'])
                full_response = f"{result['answer']}||SOURCE||{sources_str}"
            else:
                full_response = result['answer']
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": full_response
            })
            st.session_state.show_typing = False
            st.rerun()
    
    show_contact_hr_card()
    
    st.markdown("""
        <div class="footer">
            <p>🕐 Available 24/7 | 🔒 Conversations are private and secure</p>
            <p>⚠️ Answers are based on Spectron's official policy documents</p>
            <p style="font-size: 0.75rem; color: #a0aec0; margin-top: 1rem;">
                © 2025 Spectron. All rights reserved.
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
