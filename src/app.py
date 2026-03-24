
import streamlit as st
import os
import re
from pathlib import Path
from openai import OpenAI
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time

# Import the utility functions for GitHub interaction
from github_utils import get_github_pdf_urls, download_pdf

# ============== CONFIGURATION ==============
GITHUB_POLICY_REPO_URL = "https://github.com/HR-Chatbot/hr-policy-chatbot/tree/main/policies"
LOCAL_POLICY_DIR = Path("policies")

# Ensure local policy directory exists
LOCAL_POLICY_DIR.mkdir(exist_ok=True)

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="Spectron HR Assistant | 24/7 Policy Support",
    page_icon="💼",
    layout="wide", # Changed to wide for better UI layout
    initial_sidebar_state="expanded" # Expanded sidebar for policy navigator
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 0.5rem !important; padding-bottom: 0.5rem !important; max-width: 100% !important; }
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a365d; text-align: center; margin-bottom: 0.3rem; margin-top: 0.5rem; }
    .sub-header { font-size: 1rem; color: #4a5568; text-align: center; margin-bottom: 1rem; font-weight: 400; }
    .welcome-card { background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%); padding: 1.5rem; border-radius: 12px; color: white; margin-bottom: 1rem; box-shadow: 0 8px 20px rgba(0,0,0,0.15); }
    .welcome-title { font-size: 1.3rem; font-weight: 600; margin-bottom: 0.5rem; text-align: center; }
    .welcome-text { text-align: center; font-size: 0.95rem; line-height: 1.5; opacity: 0.95; }
    .policy-links-section { margin-bottom: 1rem; }
    .policy-links-title { font-size: 0.9rem; font-weight: 600; color: #4a5568; margin-bottom: 0.5rem; text-align: center; }
    .policy-list-container { background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.5rem; max-height: 200px; overflow-y: auto; }
    .policy-list-container::-webkit-scrollbar { width: 6px; }
    .policy-list-container::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 4px; }
    .policy-list-container::-webkit-scrollbar-thumb { background: #c53030; border-radius: 4px; }
    .policy-item { padding: 0.5rem 0.75rem; border-bottom: 1px solid #e2e8f0; font-size: 0.9rem; color: #2d3748; transition: background 0.2s ease; }
    .policy-item:hover { background: #edf2f7; }
    .policy-item:last-child { border-bottom: none; }
    .chat-container { background: #f7fafc; border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem; min-height: 150px; max-height: 400px; overflow-y: auto; }
    .chat-message { padding: 0.8rem 1rem; border-radius: 16px; margin-bottom: 0.75rem; max-width: 90%; animation: fadeIn 0.3s ease; line-height: 1.5; font-size: 0.95rem; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    .user-message { background: linear-gradient(135deg, #c53030 0%, #9b2c2c 100%); color: white; margin-left: auto; border-bottom-right-radius: 4px; box-shadow: 0 3px 10px rgba(197, 48, 48, 0.3); }
    .bot-message { background: white; color: #2d3748; margin-right: auto; border-bottom-left-radius: 4px; box-shadow: 0 3px 10px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; }
    .message-header { font-size: 0.8rem; font-weight: 600; margin-bottom: 0.3rem; opacity: 0.9; }
    .input-area { background: white; padding: 0.75rem; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.08); border: 1px solid #e2e8f0; margin-bottom: 0.5rem; }
    .stTextInput>div>div>input { border-radius: 8px; border: 2px solid #e2e8f0; padding: 0.75rem; font-size: 0.95rem; }
    .stTextInput>div>div>input:focus { border-color: #c53030; box-shadow: 0 0 0 2px rgba(197, 48, 48, 0.1); }
    .typing-indicator { display: flex; align-items: center; gap: 0.4rem; padding: 0.8rem 1rem; background: white; border-radius: 16px; border-bottom-left-radius: 4px; box-shadow: 0 3px 10px rgba(0,0,0,0.08); width: fit-content; margin-bottom: 0.75rem; }
    .typing-dot { width: 6px; height: 6px; background: #c53030; border-radius: 50%; animation: typing 1.4s infinite; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing { 0%, 60%, 100% { transform: translateY(0); } 30% { transform: translateY(-8px); } }
    .policy-answer { line-height: 1.7; }
    .policy-answer strong { color: #1a365d; }
    .decline-box { background: #fff5f5; border-left: 3px solid #c53030; padding: 0.75rem; border-radius: 6px; color: #c53030; font-size: 0.9rem; }
    .greeting-box { background: #f0fff4; border-left: 3px solid #38a169; padding: 0.75rem; border-radius: 6px; color: #2f855a; font-size: 0.9rem; }
    .contact-card { background: linear-gradient(135deg, #f6e05e 0%, #d69e2e 100%); color: #744210; padding: 1rem; border-radius: 12px; text-align: center; margin-top: 0.5rem; font-size: 0.9rem; }
    .contact-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; }
    /* Hide ALL unwanted elements */
    .stDeployButton, #MainMenu, footer, header { display: none !important; visibility: hidden !important; }
    /* Custom sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f0f2f6;
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a365d;
        margin-bottom: 1rem;
        text-align: center;
    }
    .policy-nav-item {
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.4rem;
        border-radius: 6px;
        background-color: white;
        border: 1px solid #e2e8f0;
        cursor: pointer;
        transition: background-color 0.2s ease;
        font-size: 0.9rem;
        color: #2d3748;
    }
    .policy-nav-item:hover {
        background-color: #e6f0ff;
        border-color: #a3c2ff;
    }
    .policy-nav-item.active {
        background-color: #cce0ff;
        border-color: #6699ff;
        font-weight: 500;
        color: #1a365d;
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
        'policy_files': [], # Store Path objects for local files
        'input_counter': 0,
        'last_query': '',
        'github_pdf_urls': [], # Store URLs fetched from GitHub
        'last_github_check': 0 # Timestamp of last GitHub check
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
    except Exception as e:
        st.error(f"Error extracting text from {pdf_path}: {e}")
        return ""

def chunk_text(text, chunk_size=400, overlap=80):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 40:
            chunks.append(chunk)
    return chunks

def load_policies_from_github():
    st.session_state.last_github_check = time.time() # Update check time
    github_urls = get_github_pdf_urls(GITHUB_POLICY_REPO_URL)

    if not github_urls:
        st.warning("Could not fetch policy URLs from GitHub. Please check the repository URL or your internet connection.")
        return False

    new_policy_downloaded = False
    current_local_files = {f.name for f in LOCAL_POLICY_DIR.glob("*.pdf")}
    github_filenames = {url.split('/')[-1] for url in github_urls}

    # Remove local files that are no longer on GitHub
    for local_file_name in current_local_files:
        if local_file_name not in github_filenames:
            (LOCAL_POLICY_DIR / local_file_name).unlink()
            st.info(f"Removed outdated policy: {local_file_name}")
            new_policy_downloaded = True

    # Download new or updated policies
    for url in github_urls:
        filename = url.split('/')[-1]
        local_path = LOCAL_POLICY_DIR / filename

        # Simple check: if file doesn't exist locally, download it
        # For more robust update, could compare file hashes or modification dates
        if not local_path.exists():
            st.info(f"Downloading new policy: {filename}")
            if download_pdf(url, local_path):
                new_policy_downloaded = True
            else:
                st.error(f"Failed to download {filename}")

    if new_policy_downloaded or not st.session_state.policies_loaded:
        st.session_state.policy_files = list(LOCAL_POLICY_DIR.glob("*.pdf"))
        if not st.session_state.policy_files:
            st.warning("No PDF policy files found locally after GitHub sync.")
            return False

        all_chunks, chunk_sources = [], []
        policy_texts = {}

        for pdf_file in st.session_state.policy_files:
            text = extract_text_from_pdf(pdf_file)
            if text:
                policy_texts[pdf_file.name] = text
                chunks = chunk_text(text)
                for chunk in chunks:
                    all_chunks.append(chunk)
                    chunk_sources.append(pdf_file.name)

        st.session_state.policy_chunks = all_chunks
        st.session_state.policy_sources = chunk_sources
        st.session_state.policy_texts = policy_texts
        st.session_state.vectorizer, st.session_state.tfidf_matrix = setup_vectorizer(all_chunks)
        st.session_state.policies_loaded = True
        st.success("Policies loaded/updated successfully!")
        return True
    return False

def setup_vectorizer(chunks):
    if not chunks:
        return None, None
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def search_policy(query, top_k=5):
    """Search for relevant policy chunks using TF-IDF cosine similarity."""
    if st.session_state.vectorizer is None or st.session_state.tfidf_matrix is None:
        return [], []

    query_vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, st.session_state.tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]

    relevant_chunks = []
    relevant_sources = []
    for i in top_indices:
        # Filter out chunks with very low similarity scores
        if similarities[i] > 0.1: # Threshold can be adjusted
            relevant_chunks.append(st.session_state.policy_chunks[i])
            relevant_sources.append(st.session_state.policy_sources[i])
    return relevant_chunks, relevant_sources

# ============== OPENAI SETUP ==============
def setup_openai():
    if st.session_state.openai_client is None:
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                st.error("OpenAI API key not found. Please set it in .streamlit/secrets.toml or as an environment variable.")
                return None
            st.session_state.openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {e}")
            return None
    return st.session_state.openai_client

def get_chatbot_response(query, policy_context, client):
    """Generates a response using OpenAI, strictly based on provided policy context."""
    if not client:
        return "I cannot connect to the AI assistant. Please check the OpenAI API key."

    system_prompt = """
    You are a highly professional and strict HR Policy Assistant for Spectron. 
    Your primary goal is to provide accurate and helpful information to employees 
    based *solely* on the provided HR policy documents. 
    
    Adhere to the following rules:
    1.  **STRICTLY POLICY-BASED**: You MUST ONLY use the information contained within the provided `Policy Context`. 
        DO NOT use any external knowledge, common sense, or make assumptions.
    2.  **DECLINE IF NOT FOUND**: If the answer to the employee's question cannot be found or inferred from the provided `Policy Context`, 
        you MUST respond with the exact phrase: "NOT_IN_POLICY".
    3.  **COMPREHENSIVE ANSWERS**: If information is available, provide a detailed and comprehensive answer. 
        Do not be brief if the context allows for a thorough explanation.
    4.  **STRUCTURED FORMAT**: Format your answers clearly using bullet points, numbered lists, and bold text for key terms or sections. 
        Organize the information logically to be easily understandable.
    5.  **NEUTRAL TONE**: Maintain a neutral, professional, and helpful tone.
    6.  **NO SPECULATION**: Do not speculate or offer advice beyond what is explicitly stated in the policies.
    7.  **CITE SOURCES**: Mention the policy document(s) from which the information was extracted, e.g., "(Source: [Policy Name].pdf)".
    
    Example of a good answer:
    "Based on the [Leave Policy].pdf, you may apply for Casual Leave. The procedure is as follows:
    *   Submit a leave request through the HR portal at least 3 days in advance.
    *   Ensure you have sufficient leave balance.
    *   Await approval from your direct manager.
    (Source: Leave Policy.pdf)"
    """

    user_prompt = f"""
    Employee Question: {query}
    
    Policy Context (relevant excerpts from HR documents):
    {policy_context}
    
    Please provide an answer based on the above context. If the answer is not in the context, respond with "NOT_IN_POLICY".
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo", # Using gpt-3.5-turbo as in original code, can be upgraded
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Keep temperature low for factual responses
            max_tokens=1000 # Increased max_tokens for more detailed answers
        )
        answer = response.choices[0].message.content.strip()
        if "NOT_IN_POLICY" in answer.upper() or len(answer) < 20: # Check for explicit decline or very short, unhelpful answers
            return None
        return answer
    except Exception as e:
        st.error(f"OpenAI API error: {e}")
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

# ============== UI COMPONENTS ==============
def show_logo():
    # Placeholder for logo, as per screenshot, it's part of the header
    # For now, using a simple text-based logo as in the original code's fallback
    st.markdown("""
    <div style="text-align: center; margin-bottom: 0.5rem;">
        <div style="font-size: 2rem; font-weight: 800; color: #1a365d; letter-spacing: 3px;">SPECTRON</div>
        <div style="font-size: 0.8rem; color: #4a5568;">HR ASSISTANT</div>
    </div>
    """, unsafe_allow_html=True)

def show_welcome_card():
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">👋 Welcome to Spectron HR Support</div>
        <div class="welcome-text">
            I'm here to help with any questions about company policies, benefits, leave,
            compensation, and more. Just type your question below.
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_policy_navigator():
    st.sidebar.markdown("<div class='sidebar-header'>Policy Navigator</div>", unsafe_allow_html=True)
    st.sidebar.markdown("Browse available policies or ask any question directly.")

    if not st.session_state.policy_files:
        st.sidebar.info("No policies loaded yet.")
        return

    st.sidebar.markdown("--- ")
    st.sidebar.markdown("**LEAVE & ATTENDANCE**")
    # Group policies by category if possible, for now just list all
    policy_names = []
    for pdf_file in st.session_state.policy_files:
        # Remove .pdf and replace underscores/hyphens with spaces, then title case
        name = pdf_file.name.replace('.pdf', '').replace('_', ' ').replace('-', ' ').title()
        policy_names.append(name)

    for name in sorted(policy_names):
        st.sidebar.markdown(f"<div class='policy-nav-item'>📄 {name}</div>", unsafe_allow_html=True)

def display_chat():
    """Display chat messages"""
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
                        I apologize, but I don't have information about this in our policy
                        documents. Please contact HR at <strong>hrd@spectron.in</strong> or
                        <strong>+91 22 4606 6960</strong>.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <div class="message-header">🤖 HR Assistant</div>
                    <div class="policy-answer">
                        {content}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ============== MAIN APP LOGIC ==============
def main():
    init_session_state()
    client = setup_openai()

    # Auto-update policies from GitHub every 5 minutes (300 seconds)
    if time.time() - st.session_state.last_github_check > 300 or not st.session_state.policies_loaded:
        with st.spinner("Updating policies from GitHub..."):
            load_policies_from_github()

    # UI Layout
    st.sidebar.empty() # Clear sidebar for fresh render
    with st.sidebar:
        show_policy_navigator()

    st.markdown("""
    <div style="background-color: #1a365d; padding: 0.5rem 1rem; color: white; display: flex; justify-content: space-between; align-items: center;">
        <div style="display: flex; align-items: center;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">💼</span>
            <span style="font-size: 1.2rem; font-weight: 600;">Spectron HR Assistant</span>
        </div>
        <div style="font-size: 0.8rem;">
            Need help? Contact HR: hr@spectron.com
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='main-header'>24/7 Policy Support</h1>", unsafe_allow_html=True)

    # Main chat area
    chat_placeholder = st.container()

    with chat_placeholder:
        if not st.session_state.chat_history:
            show_welcome_card()
            st.markdown("<div style='text-align: center; margin-top: 2rem; color: #4a5568;'>No messages yet. Start by asking a question!</div>", unsafe_allow_html=True)
        else:
            display_chat()

    # User input
    st.markdown("<div class='input-area'>", unsafe_allow_html=True)
    user_query = st.text_input("Ask any HR question...", key="user_input", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if user_query and client:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.session_state.input_counter += 1

        with chat_placeholder:
            display_chat() # Re-display chat with new user message
            st.markdown("""
            <div class="typing-indicator">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
            """, unsafe_allow_html=True)

        # Process query
        if is_greeting(user_query):
            bot_response = get_greeting_response()
        else:
            # Search for relevant policy chunks
            relevant_chunks, relevant_sources = search_policy(user_query, top_k=7) # Increased top_k for broader context
            
            # Combine chunks into a single context string, noting sources
            policy_context = ""
            if relevant_chunks:
                # Dedup sources and list them
                unique_sources = sorted(list(set(relevant_sources)))
                policy_context += f"Relevant policies: {', '.join(unique_sources)}\n\n"
                policy_context += "\n---\n".join(relevant_chunks)
            
            if policy_context:
                bot_response = get_chatbot_response(user_query, policy_context, client)
            else:
                bot_response = None # No relevant policy found

        if bot_response:
            st.session_state.chat_history.append({"role": "bot", "content": bot_response})
        else:
            st.session_state.chat_history.append({"role": "bot", "content": "DECLINE"})
        
        # Rerun to update chat display and remove typing indicator
        st.rerun()

if __name__ == "__main__":
    main()
