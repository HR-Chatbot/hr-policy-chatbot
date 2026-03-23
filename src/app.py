"""
HR Policy Chatbot for Spectron - Smart Suggestion Version
Correctly suggests Privilege Leave for planned long leaves
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
    
    .suggestion-highlight {
        background: #e6fffa;
        border: 2px solid #38b2ac;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.75rem 0;
    }
    
    .suggestion-title {
        font-weight: 700;
        color: #234e52;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    
    .procedure-box {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        padding: 0.75rem;
        border-radius: 0 8px 8px 0;
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

# ============== SMART POLICY DETECTION ==============
def detect_leave_type(query):
    """
    Smart detection of which leave type to suggest based on query context.
    Returns: 'privilege', 'sick', 'casual', or None
    """
    query_lower = query.lower()
    
    # Keywords indicating PLANNED leave = Privilege Leave
    planned_indicators = [
        'plan', 'planning', 'planned', 'advance', 'future', 'next month', 
        'next week', 'upcoming', 'vacation', 'holiday', 'trip', 'travel',
        'may', 'june', 'july', 'august', 'september', 'october', 
        'november', 'december', 'january', 'february', 'march', 'april',
        '2 weeks', '3 weeks', '1 week', 'weeks', 'month', 'long leave'
    ]
    
    # Keywords indicating MEDICAL/URGENT = Sick Leave
    sick_indicators = [
        'sick', 'ill', 'illness', 'medical', 'doctor', 'hospital', 
        'fever', 'health', 'unwell', 'not feeling', 'emergency',
        'medical certificate', 'mc', 'consultation'
    ]
    
    # Keywords indicating SHORT/CASUAL = Casual Leave
    casual_indicators = [
        'casual', 'personal work', 'urgent work', 'half day', 
        'few hours', 'personal', 'family function', 'short'
    ]
    
    planned_score = sum(1 for word in planned_indicators if word in query_lower)
    sick_score = sum(1 for word in sick_indicators if word in query_lower)
    casual_score = sum(1 for word in casual_indicators if word in query_lower)
    
    # Return the highest scoring type
    scores = {'privilege': planned_score, 'sick': sick_score, 'casual': casual_score}
    max_type = max(scores, key=scores.get)
    
    if scores[max_type] > 0:
        return max_type
    return None

def find_best_policy_file(query, detected_type=None):
    """
    Find the best matching policy file with smart prioritization.
    """
    query_lower = query.lower()
    query_normalized = query_lower.replace(' ', '').replace('-', '').replace('_', '')
    query_words = set(w for w in query_lower.split() if len(w) > 2)
    
    best_match = None
    best_score = 0
    
    # Priority order based on detected type
    priority_policies = []
    if detected_type == 'privilege':
        priority_policies = ['privilege', 'leave']  # PL first
    elif detected_type == 'sick':
        priority_policies = ['sick', 'leave']
    elif detected_type == 'casual':
        priority_policies = ['casual', 'leave']
    else:
        priority_policies = ['leave', 'privilege', 'sick', 'casual']
    
    for pdf_file in st.session_state.policy_files:
        filename = pdf_file.name.lower().replace('.pdf', '')
        filename_normalized = filename.replace(' ', '').replace('-', '').replace('_', '')
        
        score = 0
        
        # Priority boost for detected type
        for priority in priority_policies:
            if priority in filename:
                score += 15  # High priority
                break
        
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
        
        if score > best_score:
            best_score = score
            best_match = pdf_file.name
    
    return best_match if best_score >= 3 else None

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

def is_in_policy_content(query, chunks, scores):
    """Check if query matches policy content"""
    if not chunks:
        return False
    
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    
    if scores and scores[0] > 0.03:
        combined_text = " ".join(chunks[:2]).lower()
        matches = sum(1 for w in query_words if w in combined_text)
        if matches >= 1:
            return True
    
    all_policy_text = " ".join(st.session_state.policy_texts.values()).lower()
    keyword_matches = sum(1 for w in query_words if w in all_policy_text)
    
    if keyword_matches >= 1:
        return True
    
    return False

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

def get_smart_suggestion(query, context, detected_type, policy_name, client):
    """
    Generate a smart suggestive response based on detected leave type.
    """
    try:
        # Map detected type to full name
        type_names = {
            'privilege': 'Privilege Leave',
            'sick': 'Sick Leave',
            'casual': 'Casual Leave'
        }
        suggested_type = type_names.get(detected_type, 'the appropriate leave')
        
        system_prompt = f"""You are a STRICT HR Policy Assistant for Spectron.
The user is asking about leave. Based on analysis, you should suggest {suggested_type}.
Use ONLY the provided policy context. NEVER use external knowledge."""

        user_prompt = f"""ANALYZE THE QUERY AND SUGGEST THE CORRECT LEAVE TYPE

USER QUERY: {query}

DETECTED LEAVE TYPE: {suggested_type}
POLICY DOCUMENT: {policy_name}

POLICY CONTEXT:
{context[:2500]}

INSTRUCTIONS:
1. Start with a CLEAR SUGGESTION: "Based on our {policy_name}, you may apply for {suggested_type}..."
2. Explain WHY this is the right choice (planned leave = PL, medical = SL, etc.)
3. Provide the EXACT PROCEDURE from the policy
4. Mention ELIGIBILITY and any CONDITIONS
5. If {suggested_type} is NOT appropriate based on the policy, say: WRONG_TYPE
6. Be specific and actionable - no generic welcomes

EXAMPLE GOOD RESPONSE:
"Based on our Privilege Leave Policy, you may apply for Privilege Leave for your planned 2-week absence in May.

**Why Privilege Leave?**
This is planned leave requested in advance, which falls under Privilege Leave.

**Procedure:**
1. [Steps from policy]
2. [Steps from policy]

**Important Notes:**
- [Conditions from policy]"

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
        
        if "WRONG_TYPE" in answer.upper() or len(answer) < 30:
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
    """Show welcome screen"""
    st.markdown("""
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to Your HR Assistant</div>
            <div class="welcome-text">
                Ask me about Spectron's HR policies. I'll analyze our documents and suggest the best course of action.
            </div>
        </div>
    """, unsafe_allow_html=True)

def show_policy_links():
    """Show available policy documents"""
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
    """Display chat messages"""
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
    """Generate SMART SUGGESTIVE response with correct leave type detection"""
    
    # Check if pure greeting
    if is_greeting(query) and len(query.split()) <= 2:
        welcome_response = """Hello! 👋 

I'm your HR Policy Assistant. I can analyze Spectron's policy documents and suggest the best course of action.

**For example:**
- "I want to take 2 weeks leave in May, which leave should I apply for?"
- "How do I claim travel reimbursement?"
- "What is the notice period policy?"

Please share your specific HR question!"""
        
        st.session_state.chat_history.append({"role": "assistant", "content": welcome_response})
        st.session_state.show_typing = False
        return
    
    # STEP 1: Detect leave type from query context
    detected_type = detect_leave_type(query)
    
    # STEP 2: Find best policy file based on detected type
    specific_policy = find_best_policy_file(query, detected_type)
    
    # STEP 3: Get chunks from the best policy
    chunks = []
    scores = []
    
    if specific_policy:
        chunks, scores = get_chunks_from_specific_policy(specific_policy, query)
    
    # Fallback to general search if no specific policy found
    if not chunks:
        chunks, scores = search_policy(query)
        specific_policy = None
    
    # STEP 4: Check if content is relevant
    if not is_in_policy_content(query, chunks, scores):
        st.session_state.chat_history.append({"role": "assistant", "content": "DECLINE"})
        st.session_state.show_typing = False
        return
    
    context = "\n\n".join(chunks[:5])
    policy_name = specific_policy.replace('.pdf', '') if specific_policy else 'Policy'
    
    # STEP 5: Get smart suggestion from OpenAI
    client = st.session_state.openai_client
    if client:
        response = get_smart_suggestion(query, context, detected_type, policy_name, client)
        if response:
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            # Fallback if AI fails
            fallback = f"""Based on our analysis of your query, you may apply for **{detected_type.upper() if detected_type else 'the appropriate'} Leave**.

**Policy Reference:** {policy_name}

**Key Information from Policy:**
{context[:500]}...

For complete details, please refer to the full policy document or contact HR at hrd@spectron.in"""
            st.session_state.chat_history.append({"role": "assistant", "content": fallback})
    else:
        # No API - simple fallback
        fallback = f"""Based on our {policy_name}, you may apply for **{detected_type.upper() if detected_type else 'appropriate'} Leave**.

Please contact HR at hrd@spectron.in for detailed procedure."""
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
