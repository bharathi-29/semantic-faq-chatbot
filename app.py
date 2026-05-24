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
    page_title="WhatsApp FAQ Bot",
    page_icon="💬",
    layout="centered"
)

# -----------------------------------
# WhatsApp Style CSS
# -----------------------------------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: Arial, sans-serif;
}

.main {
    background-color: #ece5dd;
}

.chat-wrapper {
    padding-bottom: 100px;
}

.user-msg {
    background-color: #DCF8C6;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px 0;
    width: fit-content;
    max-width: 75%;
    margin-left: auto;
    color: black;
    font-size: 16px;
}

.bot-msg {
    background-color: white;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 8px 0;
    width: fit-content;
    max-width: 75%;
    margin-right: auto;
    color: black;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# Load Model
# -----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# -----------------------------------
# Load FAQ Data
# -----------------------------------
@st.cache_data
def load_faq():

    with open("semantic_faq.json", "r", encoding="utf-8") as file:
        return json.load(file)

faq_data = load_faq()

# -----------------------------------
# Prepare Questions & Answers
# -----------------------------------
questions = []
answers = []

for item in faq_data:

    for q in item["questions"]:

        questions.append(q)
        answers.append(item["answer"])

# -----------------------------------
# Cache Embeddings
# -----------------------------------
@st.cache_resource
def create_embeddings():

    return model.encode(questions)

question_embeddings = create_embeddings()

# -----------------------------------
# Get Answer
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
# Title
# -----------------------------------
st.title("💬 University FAQ Chatbot")

# -----------------------------------
# Session State
# -----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------
# Display Messages
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
# Chat Input
# -----------------------------------
user_input = st.chat_input("Type your message...")

# -----------------------------------
# Handle Input
# -----------------------------------
if user_input:

    # Add User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Generate Response
    response = get_answer(user_input)

    # Fake Typing Delay
    with st.spinner("Typing..."):
        time.sleep(1)

    # Add Bot Message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()