"""
HR Policy Chatbot for Spectron - Modern UI Version
Clean, professional interface without suggestion chips
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
    page_title="Spectron HR Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Header Section */
    .header-bar {
        background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 100%);
        padding: 1.5rem 2rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .header-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .logo-icon {
        font-size: 2.5rem;
    }
    
    .header-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }
    
    .header-subtitle {
        font-size: 0.95rem;
        opacity: 0.9;
        margin: 0;
        font-weight: 300;
    }
    
    .header-contact {
        text-align: right;
        font-size: 0.85rem;
        opacity: 0.9;
    }
    
    /* Main Layout */
    .main-layout {
        display: flex;
        height: calc(100vh - 100px);
        overflow: hidden;
    }
    
    /* Left Sidebar - Policy Navigator */
    .policy-nav {
        width: 300px;
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
        display: flex;
        flex-direction: column;
    }
    
    .nav-header {
        padding: 1.5rem;
        background: white;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .nav-title {
        font-size: 1rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .nav-subtitle {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.4;
    }
    
    .policy-list {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
    }
    
    .policy-category {
        margin-bottom: 1.5rem;
    }
    
    .category-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
        padding-left: 0.5rem;
    }
    
    .policy-item {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.875rem 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.9rem;
        color: #334155;
    }
    
    .policy-item:hover {
        border-color: #c53030;
        box-shadow: 0 2px 8px rgba(197, 48, 48, 0.1);
        transform: translateX(4px);
    }
    
    .policy-icon {
        font-size: 1.25rem;
    }
    
    /* Right Chat Area */
    .chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: white;
    }
    
    /* Welcome Banner */
    .welcome-banner {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-bottom: 1px solid #bae6fd;
        padding: 2rem;
        text-align: center;
    }
    
    .welcome-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    
    .welcome-text {
        color: #475569;
        font-size: 1rem;
        max-width: 600px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Chat Container */
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 2rem;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    
    .message-wrapper {
        display: flex;
        gap: 1rem;
        max-width: 80%;
    }
    
    .message-wrapper.user {
        margin-left: auto;
        flex-direction: row-reverse;
    }
    
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }
    
    .avatar.bot {
        background: #1e3a5f;
    }
    
    .avatar.user {
        background: #c53030;
    }
    
    .message-bubble {
        padding: 1rem 1.25rem;
        border-radius: 16px;
        line-height: 1.6;
        font-size: 0.95rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .message-wrapper.bot .message-bubble {
        background: #f1f5f9;
        color: #1e293b;
        border-bottom-left-radius: 4px;
    }
    
    .message-wrapper.user .message-bubble {
        background: #1e3a5f;
        color: white;
        border-bottom-right-radius: 4px;
    }
    
    .message-meta {
        font-size: 0.75rem;
        color: #94a3b8;
        margin-top: 0.25rem;
        padding: 0 0.5rem;
    }
    
    /* Input Area */
    .input-container {
        border-top: 1px solid #e2e8f0;
        padding: 1.5rem 2rem;
        background: white;
    }
    
    .input-wrapper {
        max-width: 900px;
        margin: 0 auto;
        position: relative;
    }
    
    .stTextInput>div>div>input {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem 3.5rem 1rem 1.25rem;
        font-size: 1rem;
        transition: all 0.2s;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #1e3a5f;
        box-shadow: 0 0 0 4px rgba(30, 58, 95, 0.1);
    }
    
    .send-button {
        position: absolute;
        right: 8px;
        top: 50%;
        transform: translateY(-50%);
        background: #c53030;
        color: white;
        border: none;
        border-radius: 8px;
        width: 40px;
        height: 40px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
    }
    
    .send-button:hover {
        background: #9b2c2c;
        transform: translateY(-50%) scale(1.05);
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem 1.25rem;
        background: #f1f5f9;
        border-radius: 16px;
        border-bottom-left-radius: 4px;
        width: fit-content;
    }
    
    .typing-dot {
        width: 8px;
        height: 8px;
        background: #1e3a5f;
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }
    
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
        30% { transform: translateY(-6px); opacity: 1; }
    }
    
    /* Policy Citation Badge */
    .citation-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.375rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.75rem;
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #94a3b8;
    }
    
    .empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .policy-nav {
            display: none;
        }
    }
</style>
""", unsafe_allow_html=True)

# ============== INITIALIZATION ==============
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key from secrets or environment"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("⚠️ OpenAI API key not found.")
        return None
    return OpenAI(api_key=api_key)

# ============== POLICY DATABASE ==============
class PolicyDatabase:
    """Handles loading and indexing of all HR policies"""
    
    def __init__(self, policy_folder="policies"):
        self.policy_folder = Path(policy_folder)
        self.policies = {}
        self.sections = []
        self.vectorizer = None
        self.vectors = None
        self.load_all_policies()
    
    def load_all_policies(self):
        """Load all PDF policies from the folder"""
        if not self.policy_folder.exists():
            st.error(f"Policy folder '{self.policy_folder}' not found!")
            return
        
        pdf_files = list(self.policy_folder.glob("*.pdf"))
        
        for pdf_path in pdf_files:
            try:
                content = self.extract_pdf_text(pdf_path)
                policy_name = pdf_path.stem.replace("_", " ").replace("-", " ")
                self.policies[policy_name] = content
                self.chunk_policy(policy_name, content)
            except Exception as e:
                st.error(f"Error loading {pdf_path}: {e}")
        
        self.build_index()
    
    def extract_pdf_text(self, pdf_path):
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_policy(self, policy_name, content, chunk_size=1000):
        sections = re.split(r'\n(?=[A-Z][A-Z\s]{2,}\n|\d+\.\s+[A-Z])', content)
        for i, section in enumerate(sections):
            if len(section.strip()) > 50:
                self.sections.append({
                    'policy_name': policy_name,
                    'content': section.strip(),
                    'section_num': i
                })
    
    def build_index(self):
        if not self.sections:
            return
        texts = [s['content'] for s in self.sections]
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
        self.vectors = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=3):
        if not self.sections or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append({**self.sections[idx], 'score': similarities[idx]})
        return results
    
    def get_categorized_policies(self):
        """Group policies by category for the sidebar"""
        categories = {
            "Leave & Attendance": [],
            "Compensation & Benefits": [],
            "Workplace & Conduct": [],
            "Employment Terms": [],
            "Other": []
        }
        
        for policy in self.policies.keys():
            p_lower = policy.lower()
            if any(x in p_lower for x in ['leave', 'attendance', 'absence']):
                categories["Leave & Attendance"].append(policy)
            elif any(x in p_lower for x in ['pay', 'bonus', 'gratuity', 'esic', 'compensation', 'benefit']):
                categories["Compensation & Benefits"].append(policy)
            elif any(x in p_lower for x in ['conduct', 'safety', 'environment', 'protection', 'harassment']):
                categories["Workplace & Conduct"].append(policy)
            elif any(x in p_lower for x in ['employment', 'probation', 'notice', 'termination']):
                categories["Employment Terms"].append(policy)
            else:
                categories["Other"].append(policy)
        
        return {k: v for k, v in categories.items() if v}

# ============== HR ASSISTANT ==============
class HRAssistant:
    def __init__(self, policy_db, client):
        self.db = policy_db
        self.client = client
    
    def generate_answer(self, query):
        if not self.client:
            return "System not configured properly. Please contact IT."
        
        results = self.db.search(query, top_k=3)
        
        if not results:
            return self._handle_no_context()
        
        context_parts = []
        cited_policies = set()
        
        for result in results:
            policy = result['policy_name']
            cited_policies.add(policy)
            context_parts.append(f"From {policy}:\n{result['content'][:1200]}")
        
        combined_context = "\n\n".join(context_parts)
        primary_policy = results[0]['policy_name']
        
        return self._create_response(query, combined_context, primary_policy, cited_policies)
    
    def _create_response(self, query, context, primary_policy, all_policies):
        system_prompt = f"""You are Spectron's HR Assistant. Answer based ONLY on provided policy context.

RULES:
1. Start with: "Based on our {primary_policy}..."
2. If multiple policies apply: "Based on our {primary_policy} and related policies..."
3. Be concise but complete
4. Include procedure steps if applicable
5. Add disclaimer for complex situations

Policies referenced: {', '.join(all_policies)}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                temperature=0.3,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content
            
            # Ensure citation
            if not any(phrase in answer for phrase in ["Based on our", "According to our"]):
                answer = f"Based on our {primary_policy}, {answer}"
            
            return answer
            
        except Exception as e:
            return f"I apologize, I'm experiencing technical difficulties. Please contact HR at hr@spectron.com."
    
    def _handle_no_context(self):
        return """I don't have specific information on that in our current policy documents.

**Please contact HR directly:**
- 📧 hr@spectron.com
- 📞 Ext. 2001
- 🏢 2nd Floor, HR Department"""

# ============== UI COMPONENTS ==============
def render_header():
    st.markdown("""
    <div class="header-bar">
        <div class="header-left">
            <div class="logo-icon">💼</div>
            <div>
                <div class="header-title">Spectron HR Assistant</div>
                <div class="header-subtitle">24/7 Policy Support</div>
            </div>
        </div>
        <div class="header-contact">
            <div>Need help? Contact HR</div>
            <div>hr@spectron.com | Ext. 2001</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(policy_db):
    st.markdown('<div class="policy-nav">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="nav-header">
        <div class="nav-title">📋 Policy Navigator</div>
        <div class="nav-subtitle">Browse available policies or ask any question directly in the chat.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="policy-list">', unsafe_allow_html=True)
    
    categories = policy_db.get_categorized_policies()
    
    for category, policies in categories.items():
        st.markdown(f'<div class="category-title">{category}</div>', unsafe_allow_html=True)
        for policy in sorted(policies):
            icon = "📄"
            if "leave" in policy.lower():
                icon = "🏖️"
            elif "pay" in policy.lower() or "bonus" in policy.lower():
                icon = "💰"
            elif "safety" in policy.lower():
                icon = "🛡️"
            elif "conduct" in policy.lower():
                icon = "⚖️"
            
            st.markdown(f"""
            <div class="policy-item" onclick="document.getElementById('chat-input').focus()">
                <span class="policy-icon">{icon}</span>
                <span>{policy}</span>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_chat_area(assistant):
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    
    # Welcome banner (only show if no messages)
    if 'messages' not in st.session_state or len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-banner">
            <div class="welcome-icon">👋</div>
            <div class="welcome-title">Welcome to Spectron HR Support</div>
            <div class="welcome-text">
                I'm here to help you with any questions about company policies, benefits, leave, 
                compensation, workplace guidelines, and more. Just type your question below.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div>No messages yet. Start by asking a question!</div>
        </div>
        """, unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="message-wrapper user">
                <div class="avatar user">👤</div>
                <div>
                    <div class="message-bubble">{msg["content"]}</div>
                    <div class="message-meta">You</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-wrapper bot">
                <div class="avatar bot">🤖</div>
                <div>
                    <div class="message-bubble">{msg["content"]}</div>
                    <div class="message-meta">HR Assistant</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown('<div class="input-wrapper">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([10, 1])
    with col1:
        user_input = st.text_input(
            "",
            key="chat_input",
            placeholder="Ask any HR question...",
            label_visibility="collapsed"
        )
    with col2:
        send_clicked = st.button("➤", key="send_btn", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process message
    if send_clicked and user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner(""):
            response = assistant.generate_answer(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

# ============== MAIN ==============
def main():
    render_header()
    
    # Initialize
    client = get_openai_client()
    policy_db = PolicyDatabase()
    assistant = HRAssistant(policy_db, client)
    
    # Layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        render_sidebar(policy_db)
    
    with col2:
        render_chat_area(assistant)

if __name__ == "__main__":
    main()
