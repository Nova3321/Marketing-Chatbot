import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import re
import random
import os
from datetime import datetime


# -----------------------
# Custom CSS for Luxury Marketing & Finance UI
# -----------------------
def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;700&display=swap');

    /* Global Styles */
    .main {
        background: #1e3a8a;
        padding: 0;
        font-family: 'Lora', serif;
    }
    .stApp {
        background: #f8fafc;
    }

    /* Header */
    .header-section {
        background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
        padding: 3.5rem 2rem;
        text-align: center;
        color: #ffffff;
        border-bottom: 2px solid #d4af37;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .header-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" opacity="0.1"><polyline points="0,50 25,25 50,50 75,25 100,50" stroke="%23d4af37" stroke-width="2" fill="none"/></svg>') repeat;
        animation: chartPulse 10s linear infinite;
    }
    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0;
        color: #d4af37;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        animation: fadeIn 1s ease-in-out;
    }
    .header-caption {
        font-size: 1.2rem;
        font-weight: 400;
        color: #ffffff;
        margin-top: 0.5rem;
        opacity: 0.9;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes chartPulse {
        0% { background-position: 0 0; }
        100% { background-position: 100px 0; }
    }
    @keyframes glowHover {
        from { box-shadow: 0 2px 6px rgba(212, 175, 55, 0.3); }
        to { box-shadow: 0 4px 12px rgba(212, 175, 55, 0.5); }
    }

    /* Sidebar Enhancements */
    .sidebar .stButton > button {
        width: 100%;
        margin-bottom: 0.75rem;
        background: #1e3a8a;
        color: #d4af37;
        border: 1px solid #d4af37;
        border-radius: 0.5rem;
        padding: 0.8rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 6px rgba(212, 175, 55, 0.3);
    }
    .sidebar .stButton > button:hover {
        background: #d4af37;
        color: #1e3a8a;
        transform: translateY(-2px);
        animation: glowHover 0.5s ease-in-out;
    }
    .sidebar .stTextArea > label {
        font-weight: 600;
        color: #1e3a8a;
        font-size: 1.1rem;
    }
    .sidebar .stButton > button:first-child {
        background: #d4af37;
        color: #1e3a8a;
        border-color: #1e3a8a;
    }
    .sidebar .stButton > button:first-child:hover {
        background: #1e3a8a;
        color: #d4af37;
    }
    .sidebar .stExpander {
        background: #ffffff;
        border-radius: 0.5rem;
        border: 1px solid #d1d5db;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Metric Styling */
    .stMetric {
        background: #ffffff;
        padding: 1.2rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Chat Messages */
    .stChatMessage {
        border-radius: 0.5rem;
        padding: 1.2rem;
        margin: 0.75rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        animation: fadeIn 0.5s ease-in-out;
    }
    .user-message {
        background: #ffffff;
        color: #1e3a8a;
        border-radius: 0.5rem 0.5rem 0.1rem 0.5rem;
        border-right: 3px solid #d4af37;
    }
    .assistant-message {
        background: #f8fafc;
        color: #1e3a8a;
        border-radius: 0.5rem 0.5rem 0.5rem 0.1rem;
        border-left: 3px solid #1e3a8a;
    }
    .timestamp {
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 0.5rem;
        text-align: right;
    }

    /* Input */
    .stChatInput input {
        border-radius: 0.5rem;
        border: 1px solid #d1d5db;
        padding: 0.9rem 1.3rem;
        font-size: 1.1rem;
        background: #ffffff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stChatInput input:focus {
        border-color: #d4af37;
        box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.2);
    }

    /* Spinner */
    .stSpinner > div {
        border: 3px solid #d4af37;
        border-top-color: transparent;
        animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Success Notification */
    .stSuccess {
        background: #f0fdf4;
        color: #15803d;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------
# Téléchargement NLTK si besoin
# -----------------------
for resource in ["punkt", "stopwords"]:
    try:
        if resource == "punkt":
            nltk.data.find("tokenizers/punkt")
        else:
            nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download(resource)

# -----------------------
# Initialisation
# -----------------------
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


# -----------------------
# Prétraitement
# -----------------------
def preprocess_text(text, chunk_size=2):
    text = text.replace("\r", " ")
    text = re.sub(r'\n{1,}', '. ', text)
    sentences = sent_tokenize(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) == 0:
        sentences = [s.strip() for s in re.split(r'\s{2,}', text) if s.strip()]

    blocks = []
    for i in range(0, len(sentences), chunk_size):
        block = " ".join(sentences[i:i + chunk_size]).strip()
        if block:
            blocks.append(block)

    processed_blocks = []
    for block in blocks:
        b = block.lower()
        b = re.sub(r'[^a-z0-9\s]', ' ', b)
        words = word_tokenize(b)
        filtered = [
            stemmer.stem(w) for w in words
            if w.isalnum() and w not in stop_words and len(w) > 2
        ]
        processed_blocks.append(" ".join(filtered))

    return processed_blocks, blocks


# -----------------------
# Recherche et génération
# -----------------------
def find_relevant_blocks(query, tfidf_matrix, original_blocks, vectorizer, top_k=3):
    query_words = word_tokenize(query.lower())
    processed_query = ' '.join([
        stemmer.stem(w) for w in query_words
        if w.isalnum() and w not in stop_words and len(w) > 2
    ])
    if not processed_query.strip():
        return [], 0.0
    try:
        q_vec = vectorizer.transform([processed_query])
        sims = cosine_similarity(q_vec, tfidf_matrix).flatten()
    except Exception:
        return [], 0.0
    if sims.size == 0:
        return [], 0.0
    top_idx = sims.argsort()[-top_k:][::-1]
    relevant = [original_blocks[i] for i in top_idx]
    max_score = float(sims[top_idx[0]])
    return relevant, max_score


def highlight_keywords(text, query):
    terms = [w.lower() for w in word_tokenize(query)
             if w.isalnum() and w.lower() not in stop_words and len(w) > 2]
    terms = sorted(set(terms), key=lambda x: -len(x))
    for t in terms:
        text = re.sub(rf'(?i)\b{re.escape(t)}\b', f'<strong style="color: #d4af37;">{t}</strong>', text)
    return text


def generate_response_from_blocks(query, blocks, similarity_score, threshold=0.03, max_sentences=10, max_words=400):
    if not blocks:
        return "❌ Sorry, I don’t have information about that specific topic in my knowledge base."
    sentences = sent_tokenize(" ".join(blocks))
    if not sentences:
        sentences = re.split(r'\.|\?|!', " ".join(blocks))
        sentences = [s.strip() for s in sentences if s.strip()]
    short_sentences = sentences[:max_sentences]
    short_response = " ".join(short_sentences).strip()
    words = short_response.split()
    if len(words) > max_words:
        short_response = " ".join(words[:max_words]) + "..."
    short_response = highlight_keywords(short_response, query)
    outros = [
        " 📈 Want a case study?",
        " 💰 Critical for ROI.",
        " 📊 Need strategic insights?"
    ]
    return short_response + random.choice(outros)


def answer_query(query, vectorizer, tfidf_matrix, original_blocks):
    relevant_blocks, score = find_relevant_blocks(query, tfidf_matrix, original_blocks, vectorizer, top_k=3)
    return generate_response_from_blocks(query, relevant_blocks, score)


# -----------------------
# Streamlit app
# -----------------------
def main():
    st.set_page_config(page_title="Marketing & Finance Advisor", page_icon="📈", layout="wide")

    # Inject Custom CSS
    inject_custom_css()

    # Header Section
    st.markdown("""
    <div class="header-section">
        <h1 class="header-title">📈 Marketing & Finance Advisor</h1>
        <p class="header-caption">Unlock strategic insights for marketing and financial success.</p>
    </div>
    """, unsafe_allow_html=True)

    # Main Content with Columns
    col1, col2 = st.columns([1, 3], gap="medium")

    # Left Column for Stats
    with col1:
        st.markdown("### 💼 Market Metrics")
        st.markdown('<div class="stMetric">📊 100+ strategic insights</div>', unsafe_allow_html=True)

    # Right Column for Chat
    with col2:
        # Chat Interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for m in st.session_state.messages:
            with st.chat_message(m["role"], avatar="💼" if m["role"] == "user" else "📊"):
                timestamp = datetime.now().strftime("%H:%M")
                if m["role"] == "user":
                    st.markdown(
                        f'<div class="user-message">{m["content"]}<div class="timestamp">{timestamp}</div></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="assistant-message">{m["content"]}<div class="timestamp">{timestamp}</div></div>',
                        unsafe_allow_html=True)

        if prompt := st.chat_input("💬 Ask about marketing or finance..."):
            st.chat_message("user", avatar="💼").markdown(
                f'<div class="user-message">{prompt}<div class="timestamp">{datetime.now().strftime("%H:%M")}</div></div>',
                unsafe_allow_html=True)
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner("📊 Analyzing strategies..."):
                resp = answer_query(prompt, st.session_state.vectorizer, st.session_state.tfidf_matrix,
                                    st.session_state.original_blocks)
            st.chat_message("assistant", avatar="📊").markdown(
                f'<div class="assistant-message">{resp}<div class="timestamp">{datetime.now().strftime("%H:%M")}</div></div>',
                unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": resp})

    # Sidebar
    with st.sidebar:
        st.header("📚 Strategy Hub")
        with st.expander("💡 Key Questions", expanded=True):
            example_questions = [
                "What is marketing?",
                "What are the 4Ps of marketing?",
                "Explain market segmentation",
                "What is digital marketing?",
                "How does SEO work?",
                "What is ROI?",
                "What is financial forecasting?"
            ]
            for q in example_questions:
                if st.button(q, key=f"strategy_{q}"):
                    st.session_state.messages.append({"role": "user", "content": q})
                    resp = answer_query(q, st.session_state.vectorizer, st.session_state.tfidf_matrix,
                                        st.session_state.original_blocks)
                    st.session_state.messages.append({"role": "assistant", "content": resp})
                    st.rerun()

        st.divider()
        # Knowledge Base Stats
        if 'text' in locals():
            word_count = len(text.split())
            st.markdown(f'<div class="stMetric">📖 Knowledge Base: {word_count} insights</div>', unsafe_allow_html=True)

        st.divider()
        st.subheader("📂 Expand Strategies")
        new_text = st.text_area("Add new insights:", height=100, placeholder="Share marketing or finance strategies...")
        if st.button("📈 Update Base", key="update_base"):
            if new_text.strip():
                with open("base.txt", "a", encoding="utf-8") as f:
                    f.write("\n" + new_text.strip() + "\n")
                st.success("✅ Strategy added! Refresh to apply.")
                st.rerun()

    # Load and Preprocess on Startup
    if not os.path.exists("base.txt"):
        st.error("⚠️ 'base.txt' not found. Initialize with marketing and finance knowledge.")
        return
    with open("base.txt", "r", encoding="utf-8") as f:
        text = f.read()

    if 'vectorizer' not in st.session_state:
        with st.spinner("📂 Building knowledge base..."):
            processed_blocks, original_blocks = preprocess_text(text, chunk_size=2)
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(processed_blocks)
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
            st.session_state.original_blocks = original_blocks
            st.session_state.text = text  # Cache text for metric


if __name__ == "__main__":
    main()