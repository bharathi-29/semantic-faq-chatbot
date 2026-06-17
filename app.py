import streamlit as st
import json
import numpy as np
import re
import time

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# Page Config
# -----------------------------------
st.set_page_config(
    page_title="University FAQ Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------------
# DARK THEME CSS
# -----------------------------------
st.markdown("""
<style>

/* Entire App */
.stApp {
    background-color: #0b141a;
    color: white;
    font-family: Arial, sans-serif;
}

/* Main background */
.main {
    background-color: #0b141a;
}

/* Remove white top spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 6rem;
    max-width: 900px;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: white;
    margin-bottom: 30px;
}

/* Chat wrapper */
.chat-wrapper {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding-bottom: 100px;
}

/* USER MESSAGE */
.user-msg {
    background-color: #005c4b;
    color: white;
    padding: 12px 16px;
    border-radius: 14px;
    width: fit-content;
    max-width: 75%;
    margin-left: auto;
    font-size: 16px;
    line-height: 1.5;
    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
}

/* BOT MESSAGE */
.bot-msg {
    background-color: #202c33;
    color: white;
    padding: 12px 16px;
    border-radius: 14px;
    width: fit-content;
    max-width: 75%;
    margin-right: auto;
    font-size: 16px;
    line-height: 1.5;
    box-shadow: 0 2px 5px rgba(0,0,0,0.3);
}

/* Chat input container */
.stChatInputContainer {
    background-color: #111b21 !important;
    border-top: 1px solid #222 !important;
    padding: 10px !important;
}

/* Input box */
textarea {
    color: white !important;
    background-color: #202c33 !important;
}

/* Placeholder */
textarea::placeholder {
    color: #aaaaaa !important;
}

/* Hide Streamlit Footer */
footer {
    visibility: hidden;
}

/* Hide Streamlit Menu */
#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------------
# LOAD FAQ DATA
# -----------------------------------
@st.cache_data
def load_faq():
    with open("semantic_faq.json", "r", encoding="utf-8") as file:
        return json.load(file)

faq_data = load_faq()

# -----------------------------------
# PREPARE QUESTIONS & ANSWERS
# -----------------------------------
questions = []
answers = []

for item in faq_data:

    for q in item["questions"]:

        questions.append(q)
        answers.append(item["answer"])

# -----------------------------------
# CREATE EMBEDDINGS
# -----------------------------------
@st.cache_resource
def create_embeddings():
    return model.encode(questions)

question_embeddings = create_embeddings()

# -----------------------------------
# GET ANSWER FUNCTION
# -----------------------------------
def get_answer(user_input, threshold=0.5):

    user_input = user_input.lower().strip()

    user_input = re.sub(r"[^\w\s]", "", user_input)

    user_embedding = model.encode([user_input])

    similarities = cosine_similarity(
        user_embedding,
        question_embeddings
    )

    best_match_idx = np.argmax(similarities)

    best_score = similarities[0][best_match_idx]

    if best_score < threshold:
        return "Sorry, I don't have information about that."

    return answers[best_match_idx]

# -----------------------------------
# TITLE
# -----------------------------------
st.markdown(
    """
    <div class="title">
        💬 University FAQ Chatbot
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------
# SESSION STATE
# -----------------------------------
if "messages" not in st.session_state:

    st.session_state.messages = []

    # Initial bot message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! How can I help you?"
    })

# -----------------------------------
# DISPLAY CHAT
# -----------------------------------
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)

for msg in st.session_state.messages:

    if msg["role"] == "user":

        st.markdown(
            f"""
            <div class="user-msg">
                {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="bot-msg">
                {msg["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------
# CHAT INPUT
# -----------------------------------
user_input = st.chat_input("Type your message...")

# -----------------------------------
# HANDLE USER INPUT
# -----------------------------------
if user_input:

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate response
    response = get_answer(user_input)

    # Typing animation
    with st.spinner("Typing..."):
        time.sleep(1)

    # Add bot response
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()