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
    
    .faq-section {
        margin-bottom: 1rem;
    }
    
    .faq-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    
    .faq-button {
        background: #edf2f7;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
        width: 100%;
    }
    
    .faq-button:hover {
        background: #e2e8f0;
        border-color: #c53030;
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
        margin-bottom: 0.3rem;
    }
    
    .footer-text {
        text-align: center;
        font-size: 0.75rem;
        color: #a0aec0;
        margin-top: 1rem;
    }
    
    .source-tag {
        display: inline-block;
        background: #e2e8f0;
        color: #4a5568;
        font-size: 0.7rem;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ============== INITIALIZATION ==============

@st.cache_resource
def initialize_openai():
    """Initialize OpenAI client with API key from environment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("⚠️ OPENAI_API_KEY not found in environment variables")
        st.stop()
    return OpenAI(api_key=api_key)

@st.cache_data
def load_policy_documents():
    """
    Load and process all PDF policy documents from the policies directory.
    Returns: dict with 'chunks', 'vectorizer', 'tfidf_matrix', and 'metadata'
    """
    policy_dir = Path("policies")
    
    if not policy_dir.exists():
        st.error("⚠️ 'policies' directory not found. Please create it and add PDF files.")
        return None
    
    pdf_files = list(policy_dir.glob("*.pdf"))
    
    if not pdf_files:
        st.error("⚠️ No PDF files found in 'policies' directory")
        return None
    
    all_chunks = []
    all_metadata = []
    
    for pdf_path in pdf_files:
        try:
            reader = PdfReader(str(pdf_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Clean text
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Chunk by paragraphs (approx 500 chars with overlap)
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            
            for i, para in enumerate(paragraphs):
                all_chunks.append(para)
                all_metadata.append({
                    'source': pdf_path.name,
                    'chunk_id': i,
                    'preview': para[:100] + "..."
                })
                
        except Exception as e:
            st.warning(f"Error reading {pdf_path}: {e}")
    
    if not all_chunks:
        st.error("⚠️ No valid content extracted from PDFs")
        return None
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(all_chunks)
    
    return {
        'chunks': all_chunks,
        'metadata': all_metadata,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix
    }

def get_relevant_context(query, policy_data, top_k=3, threshold=0.15):
    """
    Retrieve relevant policy chunks using cosine similarity.
    Returns: list of relevant chunks or empty list if below threshold
    """
    query_vec = policy_data['vectorizer'].transform([query])
    similarities = cosine_similarity(query_vec, policy_data['tfidf_matrix']).flatten()
    
    # Get top matches above threshold
    top_indices = similarities.argsort()[-top_k:][::-1]
    relevant_chunks = []
    
    for idx in top_indices:
        if similarities[idx] > threshold:
            relevant_chunks.append({
                'text': policy_data['chunks'][idx],
                'metadata': policy_data['metadata'][idx],
                'score': similarities[idx]
            })
    
    return relevant_chunks

def is_greeting(query):
    """Check if query is a simple greeting"""
    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 
                 'good evening', 'howdy', 'greetings', 'what\'s up']
    query_clean = query.lower().strip().rstrip('!.?')
    return any(query_clean == g or query_clean.startswith(g + ' ') for g in greetings)

def is_policy_related(query, relevant_chunks):
    """
    Determine if query is answerable from policy documents.
    Uses chunk relevance and keyword heuristics.
    """
    if not relevant_chunks:
        return False
    
    # If top match has good score, likely policy-related
    if relevant_chunks[0]['score'] > 0.25:
        return True
    
    # HR policy keywords
    hr_keywords = [
        'leave', 'vacation', 'holiday', 'sick', 'pto', 'time off',
        'benefit', 'insurance', 'health', 'dental', '401k', 'retirement',
        'salary', 'pay', 'compensation', 'bonus', 'raise', 'promotion',
        'policy', 'procedure', 'guideline', 'rule', 'regulation',
        'work', 'hours', 'overtime', 'remote', 'hybrid', 'wfh',
        'employee', 'hr', 'human resources', 'manager', 'supervisor',
        'termination', 'resignation', 'notice', 'probation',
        'training', 'development', 'performance', 'review', 'evaluation',
        'harassment', 'discrimination', 'complaint', 'grievance',
        'expense', 'reimbursement', 'travel', 'allowance'
    ]
    
    query_lower = query.lower()
    keyword_matches = sum(1 for kw in hr_keywords if kw in query_lower)
    
    return keyword_matches >= 1 and relevant_chunks[0]['score'] > 0.12

def generate_response(query, relevant_chunks, client):
    """
    Generate response using OpenAI API with strict policy-only instructions.
    """
    context = "\n\n".join([
        f"[Source: {chunk['metadata']['source']}]\n{chunk['text']}"
        for chunk in relevant_chunks[:2]  # Use top 2 chunks
    ])
    
    system_prompt = """You are Spectron's HR Policy Assistant. Your role is to:
1. Answer questions STRICTLY based on the provided policy documents
2. Be concise, professional, and helpful
3. If the policy documents don't contain the answer, say you cannot answer
4. Never make up information not in the policies
5. Always cite the source document name when providing information
6. Format answers clearly with bullet points or numbered lists when appropriate"""

    user_prompt = f"""Policy Context:
{context}

User Question: {query}

Instructions: Answer based ONLY on the policy context above. If the answer isn't in the context, respond with "POLICY_NOT_FOUND". Keep the answer concise (2-4 sentences max). Cite the source document."""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=300
        )
        
        answer = response.choices[0].message.content.strip()
        
        if "POLICY_NOT_FOUND" in answer:
            return None
        
        return answer
        
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def get_decline_message():
    """Return polite decline message for non-policy questions"""
    return """I apologize, but I can only answer questions related to Spectron's HR policies and procedures. 

For questions outside my scope, please contact:
• **HR Department**: hr@spectron.com | ext. 5500
• **IT Support**: itsupport@spectron.com | ext. 8888
• **General Inquiries**: info@spectron.com"""

def get_greeting_response():
    """Return greeting response"""
    return """Hello! 👋 Welcome to the Spectron HR Policy Assistant.

I'm here to help you with questions about:
• Leave policies (vacation, sick leave, PTO)
• Benefits and compensation
• Workplace procedures
• Employee guidelines
• And other HR-related policies

How can I assist you today?"""

# ============== SESSION STATE ==============

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'policy_data' not in st.session_state:
    with st.spinner("📚 Loading policy documents..."):
        st.session_state.policy_data = load_policy_documents()
    
if 'client' not in st.session_state:
    st.session_state.client = initialize_openai()

# ============== UI COMPONENTS ==============

# Header
st.markdown('<div class="main-header">💼 Spectron HR Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your 24/7 Policy Support Companion</div>', unsafe_allow_html=True)

# Welcome Card (only show if no messages yet)
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">👋 Welcome to Spectron HR Support</div>
        <div class="welcome-text">
            I provide instant answers based on official company policies.<br>
            Ask me about leave, benefits, compensation, or any HR procedures.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # FAQ Buttons
    st.markdown('<div class="faq-section">', unsafe_allow_html=True)
    st.markdown('<div class="faq-title">🔥 Frequently Asked Questions</div>', unsafe_allow_html=True)
    
    faq_cols = st.columns(2)
    faq_questions = [
        "How many vacation days do I get?",
        "What is the remote work policy?",
        "How do I file an expense report?",
        "What are the health benefits?"
    ]
    
    for i, question in enumerate(faq_questions):
        with faq_cols[i % 2]:
            if st.button(question, key=f"faq_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Chat Container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <div class="message-header">You</div>
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        if message.get("type") == "greeting":
            css_class = "greeting-box"
        elif message.get("type") == "decline":
            css_class = "decline-box"
        else:
            css_class = "policy-answer"
            
        st.markdown(f"""
        <div class="chat-message bot-message">
            <div class="message-header">🤖 HR Assistant</div>
            <div class="{css_class}">{message["content"]}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Process new user message
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_message = st.session_state.messages[-1]["content"]
    
    # Show typing indicator
    typing_placeholder = st.empty()
    typing_placeholder.markdown("""
    <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate response
    if is_greeting(last_message):
        response = get_greeting_response()
        msg_type = "greeting"
    else:
        # Check policy documents
        if st.session_state.policy_data:
            relevant = get_relevant_context(last_message, st.session_state.policy_data)
            
            if is_policy_related(last_message, relevant):
                answer = generate_response(last_message, relevant, st.session_state.client)
                if answer:
                    response = answer
                    msg_type = "policy"
                else:
                    response = get_decline_message()
                    msg_type = "decline"
            else:
                response = get_decline_message()
                msg_type = "decline"
        else:
            response = "⚠️ Policy documents are not loaded. Please contact HR directly."
            msg_type = "error"
    
    typing_placeholder.empty()
    
    st.session_state.messages.append({
        "role": "assistant", 
        "content": response,
        "type": msg_type
    })
    st.rerun()

# Input Area
st.markdown('<div class="input-area">', unsafe_allow_html=True)
user_input = st.text_input(
    "Ask about HR policies...",
    key="user_input",
    placeholder="e.g., How do I request time off?",
    label_visibility="collapsed"
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Send Message", use_container_width=True, type="primary"):
        if user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.rerun()

st.markdown('</div>', unsafe_allow_html=True)

# Contact Card
st.markdown("""
<div class="contact-card">
    <div class="contact-title">📞 Need Human Support?</div>
    <strong>HR Department:</strong> hr@spectron.com | ext. 5500<br>
    <strong>Hours:</strong> Mon-Fri, 9:00 AM - 5:00 PM EST
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-text">
    Responses are generated based on official Spectron policy documents.<br>
    For complex situations, always consult with HR directly.
</div>
""", unsafe_allow_html=True)

# Clear chat button
if st.session_state.messages:
    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.rerun()
