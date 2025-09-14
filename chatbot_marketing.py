# educational_marketing_chatbot.py
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
import nltk

# Forcer le téléchargement à chaque run
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
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
        text = re.sub(rf'(?i)\b{re.escape(t)}\b', f'**{t}**', text)
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
        " ✅ Would you like a real-world example?",
        " 📊 This concept is essential in marketing.",
        " 💡 Do you want me to explain further?"
    ]
    return short_response + random.choice(outros)


def answer_query(query, vectorizer, tfidf_matrix, original_blocks):
    relevant_blocks, score = find_relevant_blocks(query, tfidf_matrix, original_blocks, vectorizer, top_k=3)
    return generate_response_from_blocks(query, relevant_blocks, score)


# -----------------------
# Streamlit app
# -----------------------
def main():
    st.set_page_config(page_title="🎓 Marketing Chatbot", page_icon="🎓", layout="wide")

    # Style CSS pour améliorer le design
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(to right, #f0f2f6, #ffffff);
            font-family: 'Arial', sans-serif;
        }

        .stTitle {
            font-size:36px;
            font-weight:bold;
            color:#1f2937;
        }

        .stChatMessage {
            border-radius:12px;
            padding:12px;
            margin:8px 0;
            max-width:70%;
            color: black;  /* texte noir pour lisibilité */
            font-size: 16px;
        }

        .stChatMessage.user {
            background-color:#bfdbfe;  /* bleu clair plus lisible */
            text-align:right;
            margin-left:auto;
        }

        .stChatMessage.assistant {
            background-color:#fde68a;  /* jaune clair plus lisible */
            text-align:left;
            margin-right:auto;
        }

        button {
            background-color:#6366f1;
            color:white;
            border-radius:8px;
            padding:8px 12px;
            margin:4px 0;
            width:100%;
            font-weight:bold;
        }

        button:hover {
            background-color:#4f46e5;
            cursor:pointer;
        }

        textarea {
            border-radius:8px;
            padding:8px;
            width:100%;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🎓 Marketing Educational Assistant")
    st.caption("Ask me any question about marketing. I'll answer from the knowledge base (base.txt).")

    # Charger base.txt
    if not os.path.exists("base.txt"):
        st.error("⚠️ File 'base.txt' not found. Create it and restart.")
        return
    with open("base.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Preprocess
    if 'vectorizer' not in st.session_state:
        processed_blocks, original_blocks = preprocess_text(text, chunk_size=2)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_blocks)
        st.session_state.vectorizer = vectorizer
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.original_blocks = original_blocks

    # Sidebar
    example_questions = [
        "What is marketing?",
        "What are the 4Ps of marketing?",
        "Explain market segmentation",
        "What is digital marketing?",
        "How does SEO work?",
        "What is growth hacking?",
        "What is ROI?"
    ]
    with st.sidebar:
        st.header("📚 Knowledge Base Tools")

        with st.expander("💡 Example Questions"):
            for q in example_questions:
                if st.button(q):
                    st.session_state.messages.append({"role": "user", "content": q})
                    resp = answer_query(q, st.session_state.vectorizer, st.session_state.tfidf_matrix,
                                        st.session_state.original_blocks)
                    st.session_state.messages.append({"role": "assistant", "content": resp})
                    st.rerun()

        with st.expander("📝 Knowledge Base Stats"):
            word_count = len(text.split())
            st.metric("Words in base.txt", word_count)

        with st.expander("➕ Add Content"):
            new_text = st.text_area("Add new marketing content:")
            if st.button("Add"):
                if new_text.strip():
                    with open("base.txt", "a", encoding="utf-8") as f:
                        f.write("\n" + new_text.strip() + "\n")
                    st.success("✅ Content added! Refresh to reload.")

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        role_class = "user" if m["role"] == "user" else "assistant"
        st.markdown(f"<div class='stChatMessage {role_class}'>{m['content']}</div>", unsafe_allow_html=True)

    if prompt := st.chat_input("Ask your marketing question here..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("🔎 Searching knowledge base..."):
            resp = answer_query(prompt, st.session_state.vectorizer, st.session_state.tfidf_matrix,
                                st.session_state.original_blocks)
        st.chat_message("assistant").markdown(resp)
        st.session_state.messages.append({"role": "assistant", "content": resp})

        # Boutons rapides après réponse
        col1, col2, col3 = st.columns(3)
        if col1.button("📌 Example"):
            st.session_state.messages.append({"role": "user", "content": "Give an example"})
            st.rerun()
        if col2.button("🧩 Concept"):
            st.session_state.messages.append({"role": "user", "content": "Explain concept"})
            st.rerun()
        if col3.button("📊 More info"):
            st.session_state.messages.append({"role": "user", "content": "More info"})
            st.rerun()


if __name__ == "__main__":
    main()
