"""
HR Policy Chatbot for Spectron - STRICT RAG Version
Only answers from policies, with follow-up capability within same topic
"""

import streamlit as st
import os
from pathlib import Path
from openai import OpenAI
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time

# ============== PAGE CONFIG ==============

st.set_page_config(
    page_title="Spectron HR Assistant",
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
    
    .logo-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }
    
    .welcome-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    
    .welcome-text {
        text-align: center;
        font-size: 1rem;
        line-height: 1.5;
        opacity: 0.95;
    }
    
    .example-questions {
        background: rgba(255,255,255,0.15);
        padding: 1rem;
        border-radius: 12px;
        margin-top: 1.2rem;
    }
    
    .example-questions-title {
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-align: center;
        font-size: 0.95rem;
    }
    
    .example-question {
        display: block;
        background: rgba(255,255,255,0.95);
        color: #2d3748;
        padding: 0.6rem 1rem;
        border-radius: 8px;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: center;
    }
    
    .example-question:hover {
        background: white;
        transform: translateX(3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .chat-container {
        background: #f7fafc;
        border-radius: 16px;
        padding: 1rem;
        margin-bottom: 1rem;
        min-height: 100px;
    }
    
    .chat-message {
        padding: 0.9rem 1.2rem;
        border-radius: 18px;
        margin-bottom: 0.8rem;
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
    
    .policy-reference {
        font-size: 0.8rem;
        color: #718096;
        margin-top: 0.5rem;
        padding-top: 0.5rem;
        border-top: 1px dashed #e2e8f0;
        font-style: italic;
    }
    
    .message-header {
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
        opacity: 0.9;
    }
    
    .input-container {
        background: white;
        padding: 0.8rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 0.8rem 1rem;
        font-size: 0.95rem;
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
        padding: 0.8rem 1.2rem;
        background: white;
        border-radius: 18px;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        width: fit-content;
        margin-bottom: 0.8rem;
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
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        font-size: 0.95rem;
    }
    
    .footer {
        text-align: center;
        color: #718096;
        font-size: 0.8rem;
        margin-top: 1.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
    }
    
    /* Hide empty elements */
    .stAlert, .element-container:empty {
        display: none !important;
    }
    
    @media (max-width: 768px) {
        .main-header { font-size: 1.75rem; }
        .company-name { font-size: 1.25rem; }
        .chat-message { max-width: 95%; font-size: 0.9rem; }
        .welcome-card { padding: 1.2rem; }
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
        'processing': False,  # CRITICAL: Prevents infinite loops
        'current_topic': None,
        'current_policy_source': None,
        'example_questions': []
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
            extracted = page.extract_text()
            if extracted:
                text += extracted + " "
        return text.strip()
    except Exception as e:
        return ""

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    if len(words) < 50:
        return [text] if text else []
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def extract_policy_name(filename):
    """Extract clean policy name from filename"""
    name = filename.replace('_', ' ').replace('-', ' ')
    name = re.sub(r'\.pdf$', '', name)
    name = ' '.join(word.capitalize() for word in name.split())
    return name

def load_policies():
    policies_dir = Path("policies")
    if not policies_dir.exists():
        return [], [], []
    
    pdf_files = list(policies_dir.glob("*.pdf"))
    if not pdf_files:
        return [], [], []
    
    all_chunks, chunk_sources = [], []
    example_questions = []
    
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            chunks = chunk_text(text)
            policy_name = extract_policy_name(pdf_file.name)
            for chunk in chunks:
                all_chunks.append(chunk)
                chunk_sources.append({
                    'file': pdf_file.name,
                    'policy_name': policy_name
                })
            
            # Create example question from policy name
            if policy_name and len(example_questions) < 5:
                if "leave" in policy_name.lower():
                    example_questions.append(f"What is the {policy_name}?")
                elif "attendance" in policy_name.lower():
                    example_questions.append(f"What are the {policy_name} rules?")
                else:
                    example_questions.append(f"Tell me about {policy_name}")
    
    return all_chunks, chunk_sources, example_questions

# ============== SEARCH FUNCTIONALITY ==============

def setup_vectorizer(chunks):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def find_relevant_chunks(query, top_k=5):
    if st.session_state.vectorizer is None or not st.session_state.policy_chunks:
        return [], [], []
    
    query_vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    relevant_chunks = []
    sources = []
    scores = []
    
    for idx in top_indices:
        if similarities[idx] > 0.05:  # Threshold
            relevant_chunks.append(st.session_state.policy_chunks[idx])
            sources.append(st.session_state.policy_sources[idx])
            scores.append(similarities[idx])
    
    return relevant_chunks, sources, scores

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

def format_policy_response(policy_text, policy_name, query, client, is_follow_up=False):
    """Format policy information using OpenAI"""
    try:
        if is_follow_up:
            prompt = f"""Based ONLY on the following policy text, answer the follow-up question.
            
Policy Text (from {policy_name}):
{policy_text[:2000]}

Follow-up Question: {query}

Instructions:
1. ONLY use information from the policy text above
2. If the answer is not in the policy text, politely say it's not covered
3. Keep response professional and concise
4. Use bullet points for clarity if listing information
5. Reference the policy name at the end"""
        else:
            prompt = f"""Based ONLY on the following policy text, create a helpful summary.

Policy Text (from {policy_name}):
{policy_text[:2000]}

Instructions:
1. Create a clean, professional summary using ONLY the policy text above
2. Start with a brief definition/overview
3. Use bullet points for key information
4. Keep it concise and employee-friendly
5. Do not add any information not in the policy text
6. End with an invitation for follow-up questions
7. Format like this example:

[Brief definition]

- **Key Point 1**: [details]
- **Key Point 2**: [details]
- **Key Point 3**: [details]

If you have any specific questions or need further clarification, feel free to ask."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an HR policy assistant. ONLY use the provided policy text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=600
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return None

def get_policy_based_response(query, client):
    """Main function to get response based on policies"""
    
    # Check for greetings (no API call needed)
    greetings = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]
    if query.lower().strip() in greetings or any(query.lower().startswith(g) for g in greetings):
        return "👋 Hello! I'm your HR policy assistant. Feel free to ask me about company policies, leave rules, benefits, or any HR-related questions based on our official documents.", None
    
    # Check if this is a follow-up
    follow_up_phrases = ["explain more", "tell me more", "elaborate", "can you explain", 
                        "more details", "further", "clarify", "what about", "and", "also"]
    is_follow_up = any(phrase in query.lower() for phrase in follow_up_phrases)
    
    if is_follow_up and st.session_state.current_policy_source:
        # Use current topic's policy for follow-up
        source = st.session_state.current_policy_source
        # Get all chunks for this policy
        policy_chunks = []
        for chunk, s in zip(st.session_state.policy_chunks, st.session_state.policy_sources):
            if s['file'] == source['file']:
                policy_chunks.append(chunk)
        
        if policy_chunks:
            policy_text = " ".join(policy_chunks[:5])
            response = format_policy_response(policy_text, source['policy_name'], query, client, is_follow_up=True)
            if response:
                return response, source['policy_name']
    
    # Search for relevant policy
    relevant_chunks, sources, scores = find_relevant_chunks(query)
    
    if not relevant_chunks:
        st.session_state.current_topic = None
        st.session_state.current_policy_source = None
        return None, None
    
    # Get the best matching policy
    best_source = sources[0]
    policy_text = " ".join(relevant_chunks[:5])
    
    # Format response
    response = format_policy_response(policy_text, best_source['policy_name'], query, client, is_follow_up=False)
    
    if response:
        # Update current topic
        st.session_state.current_topic = query
        st.session_state.current_policy_source = best_source
        return response, best_source['policy_name']
    
    return None, None

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

def show_welcome_screen(example_questions):
    if not example_questions:
        example_questions = [
            "What is the leave policy?",
            "How many sick days do I get?",
            "What is the notice period?",
            "How to apply for leave?",
            "What are company working hours?"
        ]
    
    example_html = '<div class="example-questions"><div class="example-questions-title">💡 Try asking:</div>'
    for q in example_questions[:5]:
        example_html += f'<div class="example-question">{q}</div>'
    example_html += '</div>'
    
    st.markdown(f"""
        <div class="welcome-card">
            <div class="welcome-title">👋 HR Policy Assistant</div>
            <div class="welcome-text">
                Ask me about company policies, leave rules, benefits, and more.
                I'll provide answers based on official HR documents.
            </div>
            {example_html}
        </div>
    """, unsafe_allow_html=True)

def display_chat_history():
    if st.session_state.chat_history:
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
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def show_typing_indicator():
    st.markdown("""
        <div class="typing-indicator">
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <div class="typing-dot"></div>
            <span style="color: #718096; font-size: 0.9rem; margin-left: 0.5rem;">Searching policies...</span>
        </div>
    """, unsafe_allow_html=True)

def show_contact_hr_card():
    st.markdown("""
        <div class="contact-hr-card">
            <div style="font-weight: 600; margin-bottom: 0.5rem;">📞 Need personal assistance?</div>
            <div style="font-size: 0.9rem;">
                📧 hrd@spectron.in<br>
                📞 +91 22 4606 6960 EXTN: 247<br>
                🕐 Mon-Sat, 10 AM to 6 PM
            </div>
        </div>
    """, unsafe_allow_html=True)

# ============== MAIN APP ==============

def main():
    init_session_state()
    
    # Header with logo
    show_logo()
    st.markdown('<div class="sub-header">Your 24/7 HR policy assistant</div>', unsafe_allow_html=True)
    
    # Setup OpenAI
    if st.session_state.openai_client is None:
        st.session_state.openai_client = setup_openai()
    
    # Load policies silently
    if not st.session_state.policies_loaded:
        chunks, sources, example_qs = load_policies()
        if chunks:
            st.session_state.policy_chunks = chunks
            st.session_state.policy_sources = sources
            st.session_state.vectorizer, st.session_state.tfidf_matrix = setup_vectorizer(chunks)
            st.session_state.policies_loaded = True
            st.session_state.example_questions = example_qs
    
    # Show welcome screen only if no chat history
    if not st.session_state.chat_history:
        show_welcome_screen(st.session_state.example_questions)
    
    # Chat display
    display_chat_history()
    
    # Show typing indicator if processing
    if st.session_state.processing:
        show_typing_indicator()
    
    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            query = st.text_input(
                "Ask your question...",
                placeholder="Type your question here and press Enter...",
                label_visibility="collapsed",
                key="user_input"
            )
        with col2:
            submit = st.form_submit_button("Send", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Clear Chat button
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("🔄 Clear", use_container_width=True, key="clear_btn"):
            st.session_state.chat_history = []
            st.session_state.current_topic = None
            st.session_state.current_policy_source = None
            st.session_state.processing = False
            st.rerun()
    with col2:
        if st.button("📞 HR", use_container_width=True, key="contact_btn"):
            st.info("Contact HR: hrd@spectron.in | +91 22 4606 6960 EXTN: 247")
    
    # Handle submission - CRITICAL FIX for infinite loop
    if (submit or query) and query and not st.session_state.processing:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})
        st.session_state.processing = True
        st.rerun()
    
    # Process AI response - ONLY if processing flag is True
    if st.session_state.processing and st.session_state.chat_history:
        last_msg = st.session_state.chat_history[-1]
        
        # Ensure we're processing a user message
        if last_msg['role'] == 'user':
            
            # Check OpenAI client
            if not st.session_state.openai_client:
                response = "⚠️ OpenAI service not configured. Please contact HR directly at hrd@spectron.in"
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.session_state.processing = False
                st.rerun()
            else:
                # Get response based on policies
                response, policy_name = get_policy_based_response(last_msg['content'], st.session_state.openai_client)
                
                if response:
                    # Add policy reference if available
                    if policy_name:
                        response += f'\n\n<div class="policy-reference">📄 Source: {policy_name}</div>'
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    # Politely decline - no relevant policy found
                    decline_msg = """I couldn't find information about this in the company policies. 

Please contact HR directly for assistance:
📧 hrd@spectron.in
📞 +91 22 4606 6960 EXTN: 247"""
                    st.session_state.chat_history.append({"role": "assistant", "content": decline_msg})
                    st.session_state.current_topic = None
                    st.session_state.current_policy_source = None
                
                # CRITICAL: Turn off processing flag
                st.session_state.processing = False
                st.rerun()
    
    # Contact HR card
    show_contact_hr_card()
    
    # Footer
    st.markdown("""
        <div class="footer">
            <p>🕐 Available 24/7 | 🔒 Responses based on official HR policies</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
