"""
HR Policy Chatbot for Spectron
Features: Auto-policy loading, GitHub integration, Analytics, Export
"""

import streamlit as st
import os
import json
import hashlib
import time
import requests
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from collections import Counter

import pandas as pd
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

# ============== CONFIGURATION ==============
class Config:
    """Configuration settings - EDIT THESE VALUES"""
    
    # GitHub Integration (Set these to enable auto-updating policies)
    GITHUB_REPO = ""           # Format: "username/repo-name" (e.g., "spectronhr/policies")
    GITHUB_TOKEN = ""          # Leave empty for public repos, required for private repos
    
    # AI Model Settings
    GEMINI_MODEL = "gemini-2.0-flash"  # Options: "gemini-2.0-flash" (fast/cheap) or "gemini-2.5-pro" (smart/expensive)
    MAX_CONTEXT_CHUNKS = 5     # How many document sections to use per answer (3-7 recommended)
    
    # Policy Loading
    AUTO_REFRESH_HOURS = 1     # Check for new policies every X hours
    
    # Features
    ENABLE_ANALYTICS = True
    ENABLE_EXPORT = True
    ENABLE_FEEDBACK = True
    
    # UI Settings
    THEME_COLOR = "#c53030"
    SECONDARY_COLOR = "#1a365d"
    COMPANY_NAME = "SPECTRON"
    HR_EMAIL = "hrd@spectron.in"
    HR_PHONE = "+91 22 4606 6960"
    HR_EXT = "247"

# ============== CUSTOM CSS ==============
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{ font-family: 'Inter', sans-serif; }}
    
    .main-header {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {Config.SECONDARY_COLOR};
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    
    .welcome-card {{
        background: linear-gradient(135deg, {Config.SECONDARY_COLOR} 0%, #2c5282 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(0,0,0,0.15);
    }}
    
    .welcome-title {{
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-align: center;
    }}
    
    .welcome-text {{
        text-align: center;
        font-size: 1.05rem;
        line-height: 1.6;
        opacity: 0.95;
    }}
    
    .example-questions {{
        background: rgba(255,255,255,0.15);
        padding: 1.25rem;
        border-radius: 12px;
        margin-top: 1.5rem;
    }}
    
    .example-questions-title {{
        font-weight: 600;
        margin-bottom: 0.75rem;
        text-align: center;
    }}
    
    .example-question {{
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
    }}
    
    .example-question:hover {{
        background: white;
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    .chat-container {{
        background: #f7fafc;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        min-height: 300px;
        max-height: 500px;
        overflow-y: auto;
    }}
    
    .empty-state {{
        text-align: center;
        color: #a0aec0;
        padding: 3rem 1rem;
    }}
    
    .empty-state-icon {{
        font-size: 4rem;
        margin-bottom: 1rem;
    }}
    
    .chat-message {{
        padding: 1rem 1.25rem;
        border-radius: 18px;
        margin-bottom: 1rem;
        max-width: 85%;
        animation: fadeIn 0.3s ease;
        line-height: 1.5;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .user-message {{
        background: linear-gradient(135deg, {Config.THEME_COLOR} 0%, #9b2c2c 100%);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        box-shadow: 0 4px 15px rgba(197, 48, 48, 0.3);
    }}
    
    .bot-message {{
        background: white;
        color: #2d3748;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
    }}
    
    .message-header {{
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        opacity: 0.9;
    }}
    
    .input-container {{
        background: white;
        padding: 1rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }}
    
    .stTextInput>div>div>input {{
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }}
    
    .stTextInput>div>div>input:focus {{
        border-color: {Config.THEME_COLOR};
        box-shadow: 0 0 0 3px rgba(197, 48, 48, 0.1);
    }}
    
    .typing-indicator {{
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
    }}
    
    .typing-dot {{
        width: 8px;
        height: 8px;
        background: {Config.THEME_COLOR};
        border-radius: 50%;
        animation: typing 1.4s infinite;
    }}
    
    .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}
    
    @keyframes typing {{
        0%, 60%, 100% {{ transform: translateY(0); }}
        30% {{ transform: translateY(-10px); }}
    }}
    
    .contact-hr-card {{
        background: linear-gradient(135deg, #f6e05e 0%, #d69e2e 100%);
        color: #744210;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin-top: 1rem;
    }}
    
    .contact-hr-title {{
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.75rem;
    }}
    
    .contact-detail {{
        margin: 0.5rem 0;
        font-size: 1rem;
    }}
    
    .sidebar-content {{
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }}
    
    .faq-item {{
        background: #f7fafc;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        cursor: pointer;
        transition: all 0.3s ease;
        border-left: 4px solid {Config.THEME_COLOR};
        text-align: left;
        width: 100%;
        border: none;
        color: #2d3748;
        font-size: 0.95rem;
    }}
    
    .faq-item:hover {{
        background: #edf2f7;
        transform: translateX(5px);
    }}
    
    .stats-card {{
        background: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        border-top: 4px solid {Config.THEME_COLOR};
    }}
    
    .stats-number {{
        font-size: 2rem;
        font-weight: 700;
        color: {Config.THEME_COLOR};
    }}
    
    .policy-badge {{
        display: inline-block;
        background: #e2e8f0;
        color: #4a5568;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 0.25rem;
        font-weight: 500;
    }}
    
    .source-tag {{
        font-size: 0.75rem;
        color: #718096;
        margin-top: 0.5rem;
        font-style: italic;
    }}
    
    .welcome-input-container {{
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }}
    
    .back-button {{
        background: linear-gradient(135deg, #718096 0%, #4a5568 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        border: none;
        cursor: pointer;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }}
    
    .back-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    .new-chat-button {{
        background: linear-gradient(135deg, {Config.THEME_COLOR} 0%, #9b2c2c 100%);
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
    }}
    
    .new-chat-button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(197, 48, 48, 0.3);
    }}
    
    @media (max-width: 768px) {{
        .main-header {{
            font-size: 1.75rem;
        }}
        .chat-message {{
            max-width: 95%;
            font-size: 0.95rem;
        }}
        .welcome-card {{
            padding: 1.5rem;
        }}
    }}
    
    div[data-testid="stButton"] > button {{
        width: 100%;
    }}
</style>
""", unsafe_allow_html=True)

# ============== SESSION STATE ==============
def init_session_state():
    defaults = {
        'chat_history': [],
        'policy_chunks': [],
        'policy_sources': [],
        'policy_metadata': {},
        'vectorizer': None,
        'tfidf_matrix': None,
        'policies_loaded': False,
        'genai_client': None,
        'show_welcome': True,
        'pending_question': None,
        'analytics': {'questions': [], 'feedback': []},
        'last_policy_check': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============== POLICY MANAGEMENT ==============
class PolicyManager:
    """Handles loading policies from multiple sources"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path):
        """Extract text from PDF with error handling"""
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading {pdf_path}: {str(e)}")
            return ""
    
    @staticmethod
    def chunk_text(text, chunk_size=1000, overlap=200):
        """Improved chunking with sentence boundaries"""
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence.split())
            
            if current_size + sentence_size > chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                overlap_size = 0
                overlap_chunk = []
                for s in reversed(current_chunk):
                    if overlap_size + len(s.split()) > overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_size += len(s.split())
                current_chunk = overlap_chunk
                current_size = overlap_size
            
            current_chunk.append(sentence)
            current_size += sentence_size
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return [c for c in chunks if len(c) > 100]
    
    @staticmethod
    def _download_from_github():
        """Download PDFs from GitHub repository"""
        if not Config.GITHUB_REPO:
            return []
        
        repo_parts = Config.GITHUB_REPO.split('/')
        if len(repo_parts) != 2:
            raise ValueError("GITHUB_REPO must be format: username/repo-name")
        
        owner, repo = repo_parts
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/"
        
        headers = {}
        if Config.GITHUB_TOKEN:
            headers["Authorization"] = f"token {Config.GITHUB_TOKEN}"
        
        response = requests.get(api_url, headers=headers, timeout=10)
        response.raise_for_status()
        contents = response.json()
        
        downloaded_files = []
        temp_dir = Path("temp_github")
        temp_dir.mkdir(exist_ok=True)
        
        for item in contents:
            if item['type'] == 'file' and item['name'].lower().endswith('.pdf'):
                file_url = item['download_url']
                file_response = requests.get(file_url, headers=headers, timeout=30)
                file_response.raise_for_status()
                
                file_path = temp_dir / item['name']
                file_path.write_bytes(file_response.content)
                downloaded_files.append(file_path)
        
        return downloaded_files
    
    @staticmethod
    def load_all_policies():
        """Load policies from local folder and/or GitHub"""
        all_chunks = []
        chunk_sources = []
        metadata = {}
        pdf_files = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load from local
        local_dir = Path("policies")
        if local_dir.exists():
            local_pdfs = list(local_dir.glob("*.pdf"))
            pdf_files.extend(local_pdfs)
            status_text.text(f"📁 Found {len(local_pdfs)} local policies")
        
        # Load from GitHub
        if Config.GITHUB_REPO:
            status_text.text("🌐 Downloading from GitHub...")
            try:
                github_pdfs = PolicyManager._download_from_github()
                pdf_files.extend(github_pdfs)
                status_text.text(f"✅ Downloaded {len(github_pdfs)} policies from GitHub")
            except Exception as e:
                st.warning(f"⚠️ GitHub download failed: {str(e)}")
        
        if not pdf_files:
            progress_bar.empty()
            status_text.empty()
            st.error("❌ No policies found! Add PDFs to 'policies' folder or set up GitHub.")
            return [], [], {}
        
        # Process PDFs
        for idx, pdf_file in enumerate(pdf_files):
            status_text.text(f"📄 Reading: {pdf_file.name}...")
            progress_bar.progress((idx + 1) / len(pdf_files))
            
            try:
                text = PolicyManager.extract_text_from_pdf(pdf_file)
                if text:
                    chunks = PolicyManager.chunk_text(text)
                    metadata[pdf_file.name] = {
                        'chunks': len(chunks),
                        'source': 'GitHub' if 'temp_github' in str(pdf_file) else 'Local',
                        'loaded_at': datetime.now().strftime("%Y-%m-%d %H:%M")
                    }
                    
                    for chunk in chunks:
                        all_chunks.append(chunk)
                        chunk_sources.append(pdf_file.name)
                        
            except Exception as e:
                st.error(f"❌ Error reading {pdf_file.name}: {str(e)}")
        
        # Cleanup
        temp_dir = Path("temp_github")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        
        progress_bar.empty()
        status_text.empty()
        
        if metadata:
            st.success(f"✅ Loaded {len(metadata)} documents ({len(all_chunks)} sections)")
        
        return all_chunks, chunk_sources, metadata

# ============== VECTOR STORE ==============
class VectorStore:
    """Manages document embeddings and search"""
    
    @staticmethod
    def setup(chunks):
        if not chunks:
            return None, None
        vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.95
        )
        tfidf_matrix = vectorizer.fit_transform(chunks)
        return vectorizer, tfidf_matrix
    
    @staticmethod
    def search(query, vectorizer, tfidf_matrix, chunks, sources, top_k=5):
        """Enhanced search with relevance scoring"""
        if vectorizer is None or not chunks:
            return []
        
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append({
                    'chunk': chunks[idx],
                    'source': sources[idx],
                    'score': float(similarities[idx])
                })
        
        return results

# ============== AI RESPONSE ==============
class AIResponder:
    """Handles AI model interactions"""
    
    def __init__(self):
        self.client = self._setup_gemini()
    
    def _setup_gemini(self):
        """Initialize Gemini client"""
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
            if not api_key:
                return None
            return genai.Client(api_key=api_key)
        except Exception as e:
            st.error(f"AI Setup Error: {str(e)}")
            return None
    
    def generate_response(self, query, context_results, chat_history):
        """Generate contextual response"""
        if not self.client:
            return self._fallback_response()
        
        context_text = ""
        sources_used = []
        
        for i, result in enumerate(context_results[:Config.MAX_CONTEXT_CHUNKS], 1):
            context_text += f"\n[Document {i}: {result['source']}]\n{result['chunk']}\n"
            if result['source'] not in sources_used:
                sources_used.append(result['source'])
        
        history_text = ""
        for msg in chat_history[-5:]:
            role = "Employee" if msg['role'] == 'user' else "HR Assistant"
            history_text += f"{role}: {msg['content']}\n"
        
        prompt = f"""You are Spectron's HR Policy Assistant. Answer based on provided documents.

CONTEXT FROM POLICIES:
{context_text}

CONVERSATION HISTORY:
{history_text}

CURRENT QUESTION: {query}

INSTRUCTIONS:
1. Answer using ONLY the provided policy documents above
2. If answer isn't in documents, say: "I don't have specific information about that in our current policies. Please contact HR at {Config.HR_EMAIL} or extn {Config.HR_EXT}."
3. Be concise, professional, and helpful
4. Use bullet points for lists
5. Cite document names when referencing specific policies
6. For leave questions, mention the HR portal/manager process

Provide a helpful response:"""
        
        try:
            response = self.client.models.generate_content(
                model=Config.GEMINI_MODEL,
                contents=prompt,
                config={'temperature': 0.3, 'max_output_tokens': 800}
            )
            
            answer = response.text
            if sources_used and "contact HR" not in answer.lower():
                answer += f"\n\n📄 Sources: {', '.join(sources_used[:2])}"
            
            return answer
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                return "API_QUOTA_EXHAUSTED"
            return f"Error: {error_msg}"
    
    def _fallback_response(self):
        """Fallback when AI is unavailable"""
        return f"""⚠️ AI service temporarily unavailable.

Please contact HR directly:
📧 {Config.HR_EMAIL}
📞 {Config.HR_PHONE} Ext: {Config.HR_EXT}
🕐 Mon-Sat, 10 AM - 6 PM"""

# ============== ANALYTICS ==============
class Analytics:
    """Tracks usage and feedback"""
    
    @staticmethod
    def log_question(question, response_time, has_answer):
        if not Config.ENABLE_ANALYTICS:
            return
        
        st.session_state.analytics['questions'].append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'response_time': response_time,
            'has_answer': has_answer
        })
    
    @staticmethod
    def log_feedback(message_idx, feedback_type):
        if not Config.ENABLE_ANALYTICS:
            return
        
        st.session_state.analytics['feedback'].append({
            'timestamp': datetime.now().isoformat(),
            'message_idx': message_idx,
            'type': feedback_type
        })
    
    @staticmethod
    def export_chat():
        if not st.session_state.chat_history:
            return None
        
        export_data = {
            'export_time': datetime.now().isoformat(),
            'chat_history': st.session_state.chat_history,
            'analytics': st.session_state.analytics
        }
        return json.dumps(export_data, indent=2)
    
    @staticmethod
    def get_stats():
        questions = st.session_state.analytics['questions']
        if not questions:
            return {}
        
        df = pd.DataFrame(questions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return {
            'total_questions': len(questions),
            'avg_response_time': df['response_time'].mean() if 'response_time' in df else 0,
            'today_questions': len(df[df['timestamp'].dt.date == datetime.now().date()]),
            'common_topics': Counter([q['question'].split()[0] for q in questions]).most_common(5)
        }

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
            show_text_logo()
    except:
        show_text_logo()

def show_text_logo():
    """Fallback text logo"""
    st.markdown(f"""
        <div style="text-align: center; margin-bottom: 1rem;">
            <div style="font-size: 2.5rem; font-weight: 800; color: {Config.THEME_COLOR}; letter-spacing: 3px; text-transform: uppercase;">
                {Config.COMPANY_NAME}
            </div>
            <div style="color: #718096; font-size: 0.9rem;">HR Policy Assistant</div>
        </div>
    """, unsafe_allow_html=True)

def show_welcome_screen():
    """Welcome screen with input options"""
    show_logo()
    
    # Check for updates
    if st.session_state.last_policy_check:
        last_check = datetime.fromisoformat(st.session_state.last_policy_check)
        if datetime.now() - last_check > timedelta(hours=Config.AUTO_REFRESH_HOURS):
            st.info("🔄 Policies were updated recently. Refresh to load latest.")
    
    st.markdown(f"""
        <div class="welcome-card">
            <div class="welcome-title">👋 Welcome to Your HR Assistant</div>
            <div class="welcome-text">
                Get instant answers about HR policies, leave, benefits, and more. 
                Available 24/7 for your convenience!
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Quick questions
    st.markdown("### 💡 Quick Questions")
    
    cols = st.columns(2)
    quick_questions = [
        "How many casual leaves per year?",
        "What is the notice period?",
        "How do I apply for medical leave?",
        "What are working hours?",
        "How to claim reimbursement?",
        "What is the dress code?"
    ]
    
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}", use_container_width=True):
                st.session_state.pending_question = question
                st.session_state.show_welcome = False
                st.rerun()
    
    # Text input
    st.markdown('<div class="welcome-input-container">', unsafe_allow_html=True)
    st.markdown("**Or ask your own question:**")
    
    with st.form(key="welcome_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your question",
            placeholder="e.g., What is the policy for work from home?",
            label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Ask Question →", use_container_width=True, type="primary")
        
        if submitted and user_input.strip():
            st.session_state.pending_question = user_input.strip()
            st.session_state.show_welcome = False
            st.rerun()
    
    st.markdown('</div>')
    
    # Show loaded policies
    if st.session_state.policy_metadata:
        with st.expander("📚 Loaded Policies"):
            for name, meta in st.session_state.policy_metadata.items():
                source_icon = "🌐" if meta['source'] == 'GitHub' else "📁"
                st.markdown(f"{source_icon} **{name}** - {meta['chunks']} sections ({meta['source']})")

def show_chat_interface():
    """Main chat interface"""
    # Header buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back", key="back_btn", use_container_width=True):
            st.session_state.show_welcome = True
            st.rerun()
    with col3:
        if st.button("🔄 New Chat", key="new_chat_btn", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Chat history
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    if not st.session_state.chat_history:
        st.markdown("""
            <div style="text-align: center; color: #a0aec0; padding: 3rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">💬</div>
                <div>Ask your first question about HR policies</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        for idx, message in enumerate(st.session_state.chat_history):
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
                    content = f"""⚠️ Service temporarily unavailable due to high demand.

Please contact HR directly:
📧 {Config.HR_EMAIL}
📞 {Config.HR_PHONE} Ext: {Config.HR_EXT}"""
                
                st.markdown(f"""
                    <div class="chat-message bot-message">
                        <div class="message-header">🤖 HR Assistant</div>
                        {content}
                    </div>
                """, unsafe_allow_html=True)
                
                # Feedback buttons
                if Config.ENABLE_FEEDBACK and idx > 0:
                    feedback_cols = st.columns([6, 1, 1])
                    with feedback_cols[1]:
                        if st.button("👍", key=f"up_{idx}", help="Helpful"):
                            Analytics.log_feedback(idx, 'positive')
                            st.toast("Thanks for your feedback!")
                    with feedback_cols[2]:
                        if st.button("👎", key=f"down_{idx}", help="Not helpful"):
                            Analytics.log_feedback(idx, 'negative')
                            st.toast("Thanks for your feedback!")
    
    st.markdown('</div>')
    
    # Input
    user_input = st.chat_input("Type your HR question here...")
    
    if user_input:
        process_message(user_input)
    
    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        process_message(q)

def process_message(user_input):
    """Process user message and generate response"""
    if not user_input.strip():
        return
    
    start_time = time.time()
    
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Search context
    results = VectorStore.search(
        user_input,
        st.session_state.vectorizer,
        st.session_state.tfidf_matrix,
        st.session_state.policy_chunks,
        st.session_state.policy_sources
    )
    
    # Generate response
    ai = AIResponder()
    response = ai.generate_response(
        user_input,
        results,
        st.session_state.chat_history[:-1]
    )
    
    # Add response
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Log analytics
    response_time = time.time() - start_time
    has_answer = "contact HR" not in response.lower() and "don't have" not in response.lower()
    Analytics.log_question(user_input, response_time, has_answer)
    
    st.rerun()

def show_sidebar():
    """Sidebar with stats and tools"""
    with st.sidebar:
        st.markdown(f"""
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <div style="font-size: 1.5rem; font-weight: 700; color: {Config.THEME_COLOR};">{Config.COMPANY_NAME}</div>
                <div style="font-size: 0.8rem; color: #718096;">HR Assistant</div>
            </div>
        """, unsafe_allow_html=True)
        
        # New Chat
        if st.button("🔄 New Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.show_welcome = True
            st.rerun()
        
        # Stats
        if Config.ENABLE_ANALYTICS:
            stats = Analytics.get_stats()
            if stats:
                st.markdown("---")
                st.markdown("### 📊 Today's Activity")
                cols = st.columns(2)
                cols[0].metric("Questions", stats.get('today_questions', 0))
                cols[1].metric("Avg Time", f"{stats.get('avg_response_time', 0):.1f}s")
        
        # Export
        if Config.ENABLE_EXPORT and st.session_state.chat_history:
            st.markdown("---")
            export_data = Analytics.export_chat()
            if export_data:
                st.download_button(
                    "📥 Export Chat",
                    export_data,
                    file_name=f"hr_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # Contact
        st.markdown("---")
        st.markdown(f"""
            ### 📞 HR Contact
            **{Config.HR_EMAIL}**  
            **{Config.HR_PHONE}**  
            Ext: {Config.HR_EXT}  
            Mon-Sat, 10 AM - 6 PM
        """)
        
        # Admin section
        with st.expander("⚙️ Admin"):
            if st.button("🔄 Reload Policies", use_container_width=True):
                st.session_state.policies_loaded = False
                st.rerun()
            
            if st.session_state.policy_metadata:
                st.markdown("**Loaded Documents:**")
                for name, meta in st.session_state.policy_metadata.items():
                    icon = "🌐" if meta['source'] == 'GitHub' else "📁"
                    st.markdown(f"{icon} {name} ({meta['chunks']} chunks)")

# ============== MAIN ==============
def main():
    init_session_state()
    
    # Load policies
    if not st.session_state.policies_loaded:
        with st.spinner("Loading HR policies..."):
            chunks, sources, metadata = PolicyManager.load_all_policies()
            st.session_state.policy_chunks = chunks
            st.session_state.policy_sources = sources
            st.session_state.policy_metadata = metadata
            st.session_state.vectorizer, st.session_state.tfidf_matrix = VectorStore.setup(chunks)
            st.session_state.last_policy_check = datetime.now().isoformat()
            st.session_state.policies_loaded = True
    
    show_sidebar()
    
    if st.session_state.show_welcome:
        show_welcome_screen()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()
