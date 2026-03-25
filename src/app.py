"""
Spectron HR Assistant - Complete Corrected Version
Features:
- Modern two-column UI with clickable policy navigator
- Strict single-policy citation based on query detection
- GPT-3.5 Turbo for cost efficiency
- PDF policy loading from policies/ folder
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
        padding: 1rem 2rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        position: sticky;
        top: 0;
        z-index: 100;
    }
    
    .header-left { 
        display: flex; 
        align-items: center; 
        gap: 0.75rem; 
    }
    
    .logo-icon { 
        font-size: 2rem; 
    }
    
    .header-title { 
        font-size: 1.5rem; 
        font-weight: 700; 
        margin: 0; 
    }
    
    .header-subtitle { 
        font-size: 0.85rem; 
        opacity: 0.9; 
        margin: 0; 
        font-weight: 300; 
    }
    
    .header-contact {
        text-align: right;
        font-size: 0.8rem;
        opacity: 0.9;
    }
    
    /* Main Layout */
    .main-layout {
        display: flex;
        height: calc(100vh - 70px);
        overflow: hidden;
    }
    
    /* Left Sidebar - Policy Navigator */
    .policy-nav {
        width: 320px;
        background: #f8fafc;
        border-right: 1px solid #e2e8f0;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .nav-header {
        padding: 1.25rem;
        background: white;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .nav-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .nav-subtitle { 
        font-size: 0.8rem; 
        color: #64748b; 
        line-height: 1.4; 
    }
    
    .policy-list {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
    }
    
    .policy-category { 
        margin-bottom: 1.25rem; 
    }
    
    .category-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.5rem;
        padding-left: 0.5rem;
    }
    
    .policy-item {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 0.85rem;
        color: #334155;
        width: 100%;
    }
    
    .policy-item:hover {
        border-color: #c53030;
        background: #fff5f5;
        transform: translateX(3px);
    }
    
    .policy-icon { 
        font-size: 1.1rem; 
    }
    
    /* Right Chat Area */
    .chat-area {
        flex: 1;
        display: flex;
        flex-direction: column;
        background: white;
        position: relative;
    }
    
    /* Welcome Banner */
    .welcome-banner {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-bottom: 1px solid #bae6fd;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .welcome-icon { 
        font-size: 2.5rem; 
        margin-bottom: 0.75rem; 
    }
    
    .welcome-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
    
    .welcome-text {
        color: #475569;
        font-size: 0.95rem;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* Chat Container */
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 1.5rem 2rem;
        display: flex;
        flex-direction: column;
        gap: 1.25rem;
    }
    
    .message-wrapper {
        display: flex;
        gap: 0.75rem;
        max-width: 85%;
        animation: slideIn 0.3s ease;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .message-wrapper.user { 
        margin-left: auto; 
        flex-direction: row-reverse; 
    }
    
    .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    
    .avatar.bot { 
        background: #1e3a5f; 
    }
    
    .avatar.user { 
        background: #c53030; 
    }
    
    .message-content {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }
    
    .message-bubble {
        padding: 0.875rem 1.25rem;
        border-radius: 16px;
        line-height: 1.6;
        font-size: 0.9rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
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
        font-size: 0.7rem; 
        color: #94a3b8; 
        padding: 0 0.5rem; 
    }
    
    .citation-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #dbeafe;
        color: #1e40af;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        width: fit-content;
    }
    
    /* Input Area */
    .input-container {
        border-top: 1px solid #e2e8f0;
        padding: 1rem 2rem;
        background: white;
    }
    
    .input-wrapper {
        max-width: 800px;
        margin: 0 auto;
        display: flex;
        gap: 0.75rem;
    }
    
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: all 0.2s;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #1e3a5f;
        box-shadow: 0 0 0 3px rgba(30, 58, 95, 0.1);
    }
    
    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #94a3b8;
        flex: 1;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .empty-icon { 
        font-size: 3rem; 
        margin-bottom: 1rem; 
        opacity: 0.4; 
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { 
        width: 6px; 
        height: 6px; 
    }
    
    ::-webkit-scrollbar-track { 
        background: transparent; 
    }
    
    ::-webkit-scrollbar-thumb { 
        background: #cbd5e1; 
        border-radius: 3px; 
    }
    
    ::-webkit-scrollbar-thumb:hover { 
        background: #94a3b8; 
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============== INITIALIZATION ==============
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client"""
    try:
        api_key = st.secrets["OPENAI_API_KEY"]
    except:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("⚠️ Please set OPENAI_API_KEY in secrets or environment variables.")
        return None
    return OpenAI(api_key=api_key)

# ============== CITATION CONTROLLER ==============
class CitationController:
    """
    Strictly controls policy citations to ensure accuracy.
    Detects query type and enforces correct single-policy citation.
    """
    
    # Keywords that indicate specific policy types
    POLICY_KEYWORDS = {
        'Privilege Leave': ['privilege', 'planned', 'vacation', 'holiday', '2 weeks', 'two weeks', 'advance booking', 'pre-planned'],
        'Sick Leave': ['sick', 'medical', 'illness', 'doctor', 'health', 'unwell', 'fever', 'hospital'],
        'Casual Leave': ['casual', 'urgent', 'personal work', 'half day', 'sudden', 'emergency'],
        'Annual Leave': ['annual', 'yearly', 'earned leave', 'accumulated'],
        'Data Protection': ['data', 'privacy', 'gdpr', 'confidential', 'personal information'],
        'Travel Reimbursement': ['travel', 'reimbursement', 'expense', 'claim', 'trip', 'conveyance'],
        'Gratuity': ['gratuity', 'retirement benefit', 'service bonus'],
        'Payment of Bonus': ['bonus', 'performance pay', 'incentive', 'diwali bonus'],
        'ESIC': ['esic', 'insurance', 'medical claim', 'employee state insurance'],
        'Workplace Safety': ['safety', 'accident', 'hazard', 'ppe', 'security'],
        'At Will Employment': ['termination', 'resign', 'notice period', 'exit', 'separation', 'at will']
    }
    
    @classmethod
    def detect_policy_type(cls, query):
        """
        Detect which policy type the query is about based on keywords.
        Returns the most relevant policy name.
        """
        query_lower = query.lower()
        scores = {}
        
        for policy_type, keywords in cls.POLICY_KEYWORDS.items():
            score = sum(2 if kw in query_lower else 0 for kw in keywords)
            # Bonus for exact phrase matches
            for kw in keywords:
                if kw in query_lower:
                    score += 1
            if score > 0:
                scores[policy_type] = score
        
        if scores:
            return max(scores, key=scores.get)
        return None
    
    @classmethod
    def get_citation_name(cls, policy_name):
        """Format policy name for citation"""
        if not policy_name:
            return "Company Policy"
        
        # If already ends with Policy, use as-is
        if "policy" in policy_name.lower():
            return policy_name
        
        return f"{policy_name} Policy"
    
    @classmethod
    def enforce_single_citation(cls, response, correct_policy):
        """
        Remove any multiple policy references and enforce single correct citation.
        """
        correct_citation = cls.get_citation_name(correct_policy)
        
        # Patterns that indicate multiple citations
        patterns = [
            r'Based on our [^,]+(,\s*[^,]+)*\s+policies?',
            r'According to our [^,]+(,\s*[^,]+)*\s+policies?',
            r'As per our [^,]+(,\s*[^,]+)*\s+policies?',
            r'Our [^,]+(,\s*[^,]+)*\s+policies?\s+(state|indicate|specify|mention|allow)',
            r'(?:Sick Leave|Privilege Leave|Casual Leave|Annual Leave)[,\s]+(?:Sick Leave|Privilege Leave|Casual Leave|Annual Leave)[,\s]+(?:and\s+)?(?:Sick Leave|Privilege Leave|Casual Leave|Annual Leave)?\s+policies?'
        ]
        
        cleaned = response
        for pattern in patterns:
            cleaned = re.sub(pattern, f'Based on our {correct_citation}', cleaned, flags=re.IGNORECASE)
        
        # Clean up remnants
        cleaned = re.sub(r'\s+and related policies?', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r',\s*and\s+[^,]+Policy', '', cleaned, flags=re.IGNORECASE)
        
        return cleaned

# ============== POLICY DATABASE ==============
class PolicyDatabase:
    """Handles loading and indexing of all HR policies from PDFs"""
    
    def __init__(self, policy_folder="policies"):
        self.policy_folder = Path(policy_folder)
        self.policies = {}  # name -> full content
        self.sections = []  # searchable chunks
        self.vectorizer = None
        self.vectors = None
        self.load_all_policies()
    
    def load_all_policies(self):
        """Load all PDF policies from the folder"""
        if not self.policy_folder.exists():
            st.error(f"❌ Policy folder '{self.policy_folder}' not found! Please create it and add PDF files.")
            return
        
        pdf_files = list(self.policy_folder.glob("*.pdf"))
        
        if not pdf_files:
            st.warning("⚠️ No PDF policies found in the policies folder.")
            return
        
        for pdf_path in pdf_files:
            try:
                content = self.extract_pdf_text(pdf_path)
                # Clean policy name from filename
                policy_name = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
                self.policies[policy_name] = content
                self.chunk_policy(policy_name, content)
            except Exception as e:
                st.error(f"❌ Error loading {pdf_path.name}: {e}")
        
        self.build_index()
    
    def extract_pdf_text(self, pdf_path):
        """Extract text from PDF"""
        reader = PdfReader(str(pdf_path))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    
    def chunk_policy(self, policy_name, content, chunk_size=1500):
        """Split policy into searchable chunks"""
        # Split by headers (all caps or numbered sections)
        sections = re.split(r'\n(?=(?:[A-Z][A-Z\s]{2,}|(?:\d+\.|[A-Z]\.)\s+[A-Z]).{0,50}\n)', content)
        
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) > 100:  # Meaningful content only
                self.sections.append({
                    'policy_name': policy_name,
                    'content': section[:2000],  # Limit chunk size
                    'section_num': i
                })
    
    def build_index(self):
        """Build TF-IDF search index"""
        if not self.sections:
            return
        
        texts = [s['content'] for s in self.sections]
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
        self.vectors = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=3):
        """Find relevant policy sections"""
        if not self.sections or self.vectorizer is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Lower threshold for flexibility
                results.append({
                    **self.sections[idx],
                    'score': similarities[idx]
                })
        return results
    
    def get_categorized_policies(self):
        """Group policies by category for sidebar"""
        categories = {
            "🌴 Leave & Attendance": [],
            "💰 Compensation & Benefits": [],
            "🏢 Workplace & Conduct": [],
            "📋 Employment Terms": [],
            "🔒 Compliance & Legal": []
        }
        
        for policy in self.policies.keys():
            p_lower = policy.lower()
            if any(x in p_lower for x in ['leave', 'attendance', 'absence', 'vacation']):
                categories["🌴 Leave & Attendance"].append(policy)
            elif any(x in p_lower for x in ['pay', 'salary', 'bonus', 'gratuity', 'esic', 'compensation', 'benefit', 'reimbursement']):
                categories["💰 Compensation & Benefits"].append(policy)
            elif any(x in p_lower for x in ['conduct', 'safety', 'environment', 'protection', 'harassment', 'discipline']):
                categories["🏢 Workplace & Conduct"].append(policy)
            elif any(x in p_lower for x in ['employment', 'probation', 'notice', 'termination', 'resign', 'at will']):
                categories["📋 Employment Terms"].append(policy)
            else:
                categories["🔒 Compliance & Legal"].append(policy)
        
        return {k: sorted(v) for k, v in categories.items() if v}

# ============== HR ASSISTANT ==============
class HRAssistant:
    """Universal HR Q&A with strict citation control"""
    
    def __init__(self, policy_db, client):
        self.db = policy_db
        self.client = client
    
    def get_relevant_context(self, query, detected_policy):
        """
        Get context from the most relevant policy.
        Prioritizes the detected policy type.
        """
        results = self.db.search(query, top_k=5)
        
        if not results:
            return None, None
        
        # If we detected a specific policy type, prioritize it
        if detected_policy:
            for result in results:
                if detected_policy.lower() in result['policy_name'].lower():
                    return result['content'][:2500], result['policy_name']
        
        # Otherwise use top result
        return results[0]['content'][:2500], results[0]['policy_name']
    
    def generate_answer(self, query):
        """Generate answer with correct single-policy citation"""
        if not self.client:
            return "⚠️ System not configured. Please contact IT support."
        
        # Step 1: Detect what type of policy this is about
        detected_policy = CitationController.detect_policy_type(query)
        
        # Step 2: Get relevant context
        context, source_policy = self.get_relevant_context(query, detected_policy)
        
        if not context:
            return self._handle_no_context()
        
        # Step 3: Determine final citation (detected type takes priority)
        if detected_policy:
            citation_policy = detected_policy
        else:
            citation_policy = source_policy
        
        citation_name = CitationController.get_citation_name(citation_policy)
        
        # Step 4: Generate response with strict instructions
        return self._create_strict_response(query, context, citation_name)
    
    def _create_strict_response(self, query, context, citation_name):
        """Create response enforcing single correct citation"""
        
        system_prompt = f"""You are Spectron's HR Assistant. Answer based ONLY on the provided policy context.

STRICT RULES:
1. You MUST start with exactly: "Based on our {citation_name},"
2. Answer the question directly and concisely
3. Include procedure steps only if relevant
4. Add 1-2 important notes if applicable
5. NEVER mention other policy names
6. If context doesn't fully answer, say so honestly

Context:
{context}
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {query}"}
                ],
                temperature=0.2,
                max_tokens=600
            )
            
            answer = response.choices[0].message.content
            
            # Post-process to enforce citation
            answer = CitationController.enforce_single_citation(answer, citation_name)
            
            # Ensure proper citation start
            if not answer.startswith(f"Based on our {citation_name}"):
                answer = f"Based on our {citation_name}, {answer}"
            
            # Add citation badge indicator
            answer = f'<div class="citation-badge">📋 {citation_name}</div>{answer}'
            
            return answer
            
        except Exception as e:
            return f"⚠️ I apologize, I'm experiencing technical difficulties. Please contact HR at hr@spectron.com."
    
    def _handle_no_context(self):
        """Fallback when no relevant policy found"""
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
    """Render clickable policy navigator"""
    st.markdown('<div class="policy-nav">', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="nav-header">
        <div class="nav-title">📋 Policy Navigator</div>
        <div class="nav-subtitle">Click any policy to learn more, or type your question below.</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="policy-list">', unsafe_allow_html=True)
    
    categories = policy_db.get_categorized_policies()
    
    for category, policies in categories.items():
        st.markdown(f'<div class="category-title">{category}</div>', unsafe_allow_html=True)
        
        for policy in policies:
            # Determine icon based on policy name
            icon = "📄"
            p_lower = policy.lower()
            if "privilege" in p_lower:
                icon = "🏖️"
            elif "sick" in p_lower:
                icon = "🏥"
            elif "casual" in p_lower:
                icon = "⚡"
            elif "bonus" in p_lower or "pay" in p_lower:
                icon = "💰"
            elif "safety" in p_lower:
                icon = "🛡️"
            elif "conduct" in p_lower:
                icon = "⚖️"
            elif "data" in p_lower:
                icon = "🔒"
            elif "travel" in p_lower:
                icon = "✈️"
            
            # Create clickable button for each policy
            if st.button(f"{icon} {policy}", key=f"policy_{policy}", use_container_width=True):
                st.session_state.policy_clicked = policy
                st.rerun()
    
    st.markdown('</div></div>', unsafe_allow_html=True)

def render_chat_area(assistant):
    """Render chat interface"""
    st.markdown('<div class="chat-area">', unsafe_allow_html=True)
    
    # Initialize messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.policy_clicked = None
    
    # Show welcome banner if no messages
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="welcome-banner">
            <div class="welcome-icon">👋</div>
            <div class="welcome-title">Welcome to Spectron HR Support</div>
            <div class="welcome-text">
                I'm here to help you with any questions about company policies, benefits, leave, 
                compensation, and more. Just type your question below or click a policy to learn more.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display messages
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div>No messages yet. Start by asking a question or selecting a policy!</div>
        </div>
        """, unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="message-wrapper user">
                <div class="avatar user">👤</div>
                <div class="message-content">
                    <div class="message-bubble">{msg["content"]}</div>
                    <div class="message-meta">You</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="message-wrapper bot">
                <div class="avatar bot">🤖</div>
                <div class="message-content">
                    <div class="message-bubble">{msg["content"]}</div>
                    <div class="message-meta">HR Assistant</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Input area
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # Handle policy click
    default_value = ""
    if st.session_state.policy_clicked:
        default_value = f"Tell me about {st.session_state.policy_clicked}"
        st.session_state.policy_clicked = None
    
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "",
            value=default_value,
            key="chat_input",
            placeholder="Ask any HR question...",
            label_visibility="collapsed"
        )
    with col2:
        send_clicked = st.button("Send ➤", key="send_btn", use_container_width=True, type="primary")
    
    st.markdown('</div></div>', unsafe_allow_html=True)
    
    # Process message
    if send_clicked and user_input:
        # Clear default if it was from policy click
        if user_input.startswith("Tell me about"):
            st.session_state.policy_clicked = None
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking..."):
            response = assistant.generate_answer(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        st.rerun()

# ============== MAIN ==============
def main():
    render_header()
    
    # Initialize systems
    client = get_openai_client()
    policy_db = PolicyDatabase()
    assistant = HRAssistant(policy_db, client)
    
    # Create layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        render_sidebar(policy_db)
    
    with col2:
        render_chat_area(assistant)

if __name__ == "__main__":
    main()
