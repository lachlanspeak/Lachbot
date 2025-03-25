import streamlit as st
import openai
import pinecone
import pyttsx3
import speech_recognition as sr
import requests
import json

# 🎯 Load API Keys from Streamlit Secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
PINECONE_ENV = st.secrets["pinecone"]["environment"]

# 🎯 Initialize OpenAI API
openai.api_key = OPENAI_API_KEY

# 🎯 Initialize Pinecone Memory
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index_name = "ai-memory"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric="cosine")

index = pinecone.Index(index_name)

# 🎯 Streamlit UI
st.title("🤖 Lach’s Fully Autonomous AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# 🎯 AI Memory Storage
def store_knowledge(user_input, response):
    """Stores conversation history into Pinecone memory."""
    vector = get_embedding(user_input)
    index.upsert(vectors=[{"id": user_input, "values": vector, "metadata": {"response": response}}])

def retrieve_knowledge(user_input):
    """Retrieve stored AI response from Pinecone memory."""
    vector = get_embedding(user_input)  
    result = index.query(vector=vector, top_k=1, include_metadata=True)

    # ✅ Prevents KeyError by checking if response exists
    if result.matches and "response" in result.matches[0].metadata:
        return result.matches[0].metadata["response"]

    return None  # No valid memory found

# 🎯 AI Response Generation
def generate_response(user_input):
    """Generates a response using OpenAI GPT-4."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a highly intelligent AI."},
            {"role": "user", "content": user_input}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()

# 🎯 AI Embeddings for Memory
def get_embedding(text):
    """Converts text into embeddings for Pinecone storage."""
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response["data"][0]["embedding"]

# 🎯 User Input Handling
user_input = st.chat_input("Type a message...")
if user_input:
    # Check if AI remembers this question
    past_response = retrieve_knowledge(user_input)
    
    if past_response:
        reply = f"🤖 (Memory Recall): {past_response}"
    else:
        reply = generate_response(user_input)
        store_knowledge(user_input, reply)  # Store new response in memory

    # Display AI response
    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(reply)

    # Store in session history
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": reply})

