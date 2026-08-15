import streamlit as st
import os
from dotenv import load_dotenv
from app_backend import GeminiChatEngine

# Force load variables in frontend too
load_dotenv(override=True)

# 1. Page Configuration
st.set_page_config(
    page_title="Full Stack Groq Chatbot",
    page_icon="⚡",
    layout="centered"
)

st.title("⚡ Full Stack Groq AI")
st.caption("Unit IV Production Chatbot Demo | Built with Streamlit & Groq")

# 2. Initialize Secrets & Engine for GROQ
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        api_key = None

if not api_key:
    st.error("⚠️ GROQ_API_KEY not found in .env! Please add it.")
    st.stop()

# Cache the engine so it doesn't re-initialize on every chat message
@st.cache_resource
def get_engine(key: str):
    return GeminiChatEngine(api_key=key)

engine = get_engine(api_key)

# 3. Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_tokens_used" not in st.session_state:
    st.session_state.total_tokens_used = 0

# Sidebar Metrics & Observability
with st.sidebar:
    st.header("📊 Session Metrics")
    st.metric(label="Total Tokens Used", value=st.session_state.total_tokens_used)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.total_tokens_used = 0
        st.rerun()

# 4. Display Existing Chat Messages
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "⚡"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 5. Handle User Input
user_input = st.chat_input("Ask Groq anything...")

if user_input:
    if len(user_input.strip()) == 0:
        st.warning("Message cannot be empty.")
        st.stop()

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    with st.chat_message("assistant", avatar="⚡"):
        with st.spinner("Groq is thinking..."):
            reply_text, token_stats = engine.generate_chat_response(
                message_history=st.session_state.messages,
                user_message=user_input
            )
            st.markdown(reply_text)

    # Update State
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": reply_text})
    st.session_state.total_tokens_used += token_stats.get("total_tokens", 0)
    
    st.rerun()