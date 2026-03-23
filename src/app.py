"""
HR Policy Chatbot for Spectron - Universal Policy Q&A System
Handles ANY employee questions about ANY HR policy with correct source attribution
"""

import streamlit as st
import os
import re
from pathlib import Path
from openai import OpenAI
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

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
        cursor: pointer;
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
    
    .source-citation {
        background: #ebf8ff;
        border-left: 3px solid #3182ce;
        padding: 0.5rem 0.75rem;
        margin-top: 0.75rem;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #2c5282;
        font-style: italic;
    }
    
    .quick-questions {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.5rem;
        justify-content: center;
    }
    
    .quick-chip {
        background: #edf2f7;
        border: 1px solid #e2e8f0;
        padding: 0.4rem 0.8rem;
        border-radius: 16px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
        color: #4a5568;
    }
    
    .quick-chip:hover {
        background: #c53030;
        color: white;
        border-color: #c53030;
    }
    
    .disclaimer {
        font-size: 0.8rem;
        color: #718096;
        text-align: center;
        margin-top: 0.5rem;
        padding: 0.5rem;
        border-top: 1px solid #e2e8f0;
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
        st.error("⚠️ OpenAI API key not found. Please set it in secrets or environment variables.")
        return None
    return OpenAI(api_key=api_key)

# ============== POLICY LOADING & INDEXING ==============
class PolicyDatabase:
    """Handles loading and indexing of all HR policies"""
    
    def __init__(self, policy_folder="policies"):
        self.policy_folder = Path(policy_folder)
        self.policies = {}  # name -> content
        self.sections = []  # list of (policy_name, section_text, metadata)
        self.vectorizer = None
        self.vectors = None
        self.load_all_policies()
    
    def load_all_policies(self):
        """Load all PDF policies from the folder"""
        if not self.policy_folder.exists():
            st.error(f"Policy folder '{self.policy_folder}' not found!")
            return
        
        pdf_files = list(self.policy_folder.glob("*.pdf"))
        
        if not pdf_files:
            st.warning("No PDF policies found in the policies folder.")
            return
        
        for pdf_path in pdf_files:
            try:
                content = self.extract_pdf_text(pdf_path)
                policy_name = pdf_path.stem.replace("_", " ").replace("-", " ")
                self.policies[policy_name] = content
                
                # Split into chunks for better retrieval
                self.chunk_policy(policy_name, content)
                
            except Exception as e:
                st.error(f"Error loading {pdf_path}: {e}")
        
        # Build search index
        self.build_index()
    
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF"""
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def chunk_policy(self, policy_name, content, chunk_size=1000, overlap=200):
        """Split policy into overlapping chunks for better retrieval"""
        # Split by sections (common headers)
        section_pattern = r'\n(?=[A-Z][A-Z\s]{2,}\n|\d+\.\s+[A-Z])'
        raw_sections = re.split(section_pattern, content)
        
        for i, section in enumerate(raw_sections):
            if len(section.strip()) > 50:  # Ignore tiny fragments
                self.sections.append({
                    'policy_name': policy_name,
                    'content': section.strip(),
                    'section_num': i,
                    'char_start': sum(len(s) for s in raw_sections[:i])
                })
    
    def build_index(self):
        """Build TF-IDF index for semantic search"""
        if not self.sections:
            return
        
        texts = [s['content'] for s in self.sections]
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.vectors = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=3):
        """Find most relevant policy sections for any query"""
        if not self.sections or self.vectorizer is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        
        # Get top matches
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                results.append({
                    **self.sections[idx],
                    'score': similarities[idx]
                })
        
        return results
    
    def get_policy_summary(self):
        """Return list of available policies"""
        return list(self.policies.keys())

# ============== UNIVERSAL RESPONSE GENERATOR ==============
class HRAssistant:
    """Universal HR Q&A system that handles any policy question"""
    
    def __init__(self, policy_db, client):
        self.db = policy_db
        self.client = client
        
        # Policy category hints for better context understanding
        self.category_hints = {
            'leave': ['leave', 'vacation', 'absence', 'time off', 'sick', 'privilege', 'casual', 'annual'],
            'compensation': ['salary', 'pay', 'wage', 'compensation', 'bonus', 'increment', 'raise', 'overtime'],
            'conduct': ['behavior', 'misconduct', 'harassment', 'discipline', 'ethics', 'violation'],
            'benefits': ['insurance', 'medical', 'health', 'pf', 'provident fund', 'gratuity', 'retirement'],
            'attendance': ['attendance', 'punctuality', 'late', 'timing', 'hours', 'work hours'],
            'recruitment': ['hiring', 'interview', 'recruitment', 'onboarding', 'joining', 'offer'],
            'termination': ['resign', 'termination', 'firing', 'notice period', 'exit', 'separation'],
            'performance': ['appraisal', 'performance', 'review', 'pms', 'rating', 'promotion'],
            'training': ['training', 'learning', 'development', 'course', 'certification']
        }
    
    def detect_category(self, query):
        """Auto-detect which policy category the question belongs to"""
        query_lower = query.lower()
        scores = {}
        
        for category, keywords in self.category_hints.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[category] = score
        
        return max(scores, key=scores.get) if scores else 'general'
    
    def generate_answer(self, query):
        """Generate answer for ANY HR question"""
        if not self.client:
            return "System not configured properly. Please contact IT."
        
        # Search for relevant policy content
        results = self.db.search(query, top_k=3)
        
        if not results:
            return self._handle_no_context(query)
        
        # Build context from multiple policies if needed
        context_parts = []
        cited_policies = set()
        
        for result in results:
            policy = result['policy_name']
            cited_policies.add(policy)
            context_parts.append(f"--- From {policy} ---\n{result['content'][:1500]}")
        
        combined_context = "\n\n".join(context_parts)
        
        # Determine primary policy for citation
        primary_policy = results[0]['policy_name']
        
        # Generate appropriate response
        return self._create_response(query, combined_context, primary_policy, cited_policies)
    
    def _create_response(self, query, context, primary_policy, all_policies):
        """Create contextual response with proper citation"""
        
        category = self.detect_category(query)
        
        system_prompt = f"""You are Spectron's HR Assistant. Answer employee questions based ONLY on the provided policy context.

CITATION RULES:
1. ALWAYS cite the specific policy name: "Based on our {primary_policy}..."
2. If multiple policies apply, mention: "Based on our {primary_policy} and related policies..."
3. NEVER make up information not in the context
4. If the answer isn't in the context, say so clearly

RESPONSE STRUCTURE:
1. **Direct Answer** - Clear, concise response with citation
2. **Details** - Relevant specifics from policy
3. **Procedure/Steps** - If applicable, numbered steps
4. **Important Notes** - Key caveats or requirements
5. **Contact** - "For further clarification, contact HR at hr@spectron.com"

Question Category: {category}
Available Policies: {', '.join(all_policies)}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nEmployee Question: {query}\n\nProvide a helpful, accurate response citing the correct policy."}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Verify citation is present
            if not any(phrase in answer for phrase in ["Based on our", "According to our", "As per our"]):
                answer = f"Based on our {primary_policy}, {answer}"
            
            return answer
            
        except Exception as e:
            return f"I apologize, I'm experiencing technical difficulties. Please contact HR directly at hr@spectron.com. (Error: {str(e)})"
    
    def _handle_no_context(self, query):
        """Handle questions when no relevant policy is found"""
        return """I don't have specific information about that in our current policy documents. 

**Please contact HR directly:**
- Email: hr@spectron.com
- Phone: Ext. 2001
- Visit: HR Department, 2nd Floor

They'll be happy to assist you with your specific query."""

# ============== UI COMPONENTS ==============
def render_header():
    st.markdown('<div class="main-header">💼 Spectron HR Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">24/7 Support for All Your HR Questions</div>', unsafe_allow_html=True)

def render_welcome():
    st.markdown("""
    <div class="welcome-card">
        <div class="welcome-title">👋 Welcome to Spectron!</div>
        <div class="welcome-text">
            I'm your AI HR Assistant. I can help you with questions about:<br>
            • Leave Policies (Sick, Privilege, Casual, etc.)<br>
            • Compensation & Benefits<br>
            • Attendance & Work Hours<br>
            • Code of Conduct<br>
            • Recruitment & Onboarding<br>
            • And any other HR-related queries
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_quick_questions():
    st.markdown('<div style="font-size: 0.9rem; color: #4a5568; margin-bottom: 0.5rem; text-align: center;">Try asking:</div>', unsafe_allow_html=True)
    
    quick_qs = [
        "How many privilege leaves do I get?",
        "What's the notice period policy?",
        "How do I claim medical insurance?",
        "What is the dress code?",
        "How is overtime calculated?",
        "What are the working hours?",
        "How do I apply for leave?",
        "What's the probation period?"
    ]
    
    cols = st.columns(4)
    for i, q in enumerate(quick_qs):
        with cols[i % 4]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                return q
    return None

def render_policy_list(policy_db):
    st.markdown('<div class="policy-links-section">', unsafe_allow_html=True)
    st.markdown('<div class="policy-links-title">📋 Available Policy Documents</div>', unsafe_allow_html=True)
    
    policies = policy_db.get_policy_summary()
    
    st.markdown('<div class="policy-list-container">', unsafe_allow_html=True)
    for policy in policies:
        st.markdown(f'<div class="policy-item">📄 {policy}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_chat_interface():
    """Main chat interface"""
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello! I'm here to help with any HR-related questions. What would you like to know about?"
        })
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user-message">
                <div class="message-header">You</div>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <div class="message-header">HR Assistant</div>
                <div class="policy-answer">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask your question...",
            key="user_input",
            placeholder="Type any HR question here...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", use_container_width=True, type="primary")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return user_input if send_button and user_input else None

# ============== MAIN APP ==============
def main():
    render_header()
    
    # Initialize systems
    client = get_openai_client()
    policy_db = PolicyDatabase()
    assistant = HRAssistant(policy_db, client)
    
    # Layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_welcome()
        render_policy_list(policy_db)
        
        st.markdown("""
        <div class="contact-card">
            <div class="contact-title">📞 Need Human Support?</div>
            <div>HR Team: hr@spectron.com<br>Ext: 2001 | 2nd Floor</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Quick questions
        quick_q = render_quick_questions()
        
        # Chat interface
        user_message = render_chat_interface()
        
        # Process input
        if quick_q:
            user_message = quick_q
        
        if user_message:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_message})
            
            # Show typing indicator
            with st.spinner(""):
                st.markdown("""
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
                """, unsafe_allow_html=True)
                
                # Generate response
                response = assistant.generate_answer(user_message)
                
                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to update chat
            st.rerun()
    
    # Footer disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚠️ This AI assistant provides information based on company policies. 
        For complex situations or policy interpretations, please consult with HR directly.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
