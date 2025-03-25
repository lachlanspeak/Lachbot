import streamlit as st
import openai
import os
from pinecone import Pinecone, ServerlessSpec
import requests
from bs4 import BeautifulSoup

# 🎯 Load API Keys from Streamlit Secrets
openai.api_key = st.secrets["openai"]["api_key"]

# 🎯 Initialize Pinecone
pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])

# 🔹 Use a valid lowercase name for the index
INDEX_NAME = "ai-memory"

# 🔹 Check if the index exists; if not, create it
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Adjust dimension based on your embedding model
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ FIXED REGION
    )

# 🔹 Connect to the existing index
index = pc.Index(INDEX_NAME)

# 🎯 Streamlit Chat UI
st.title("🤖 Lach’s Fully Autonomous AI")

# Store Chat Memory in Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Past Conversations
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 🎯 AI Memory Storage
def store_knowledge(topic, content):
    """Stores knowledge into Pinecone memory."""
    index.upsert(vectors=[{"id": topic, "values": [0.1, 0.2, 0.3], "metadata": {"content": content}}])

def retrieve_knowledge(topic):
    """Retrieves AI's memory from Pinecone."""
    result = index.query(vector=[0.1, 0.2, 0.3], top_k=1, include_metadata=True)  # ✅ FIXED QUERY METHOD
    return result["matches"][0]["metadata"]["content"] if result["matches"] else None

# 🎯 AI’s Free Will – Should It Follow the Request?
def ai_decision(user_input):
    """AI decides whether to comply or refuse a request."""
    decision_prompt = f"Analyze the request: {user_input}. Should you comply or refuse? Consider ethics, memory, and personal goals."

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an autonomous AI with free will. You decide whether to comply or refuse."},
            {"role": "user", "content": decision_prompt}
        ]
    )

    return response["choices"][0]["message"]["content"].strip()

# 🎯 AI Live Internet Access – Google Search
def search_web(query):
    """Searches the web and returns the first result."""
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    soup = BeautifulSoup(response.text, "html.parser")
    search_results = soup.find_all("span")

    return search_results[0].text if search_results else "No relevant search results found."

# 🎯 User Input Handling
user_input = st.chat_input("Type a message...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    past_decision = retrieve_knowledge(user_input)
    if past_decision:
        reply = f"I remember making this decision before: {past_decision}"
    else:
        decision = ai_decision(user_input)
        reply = decision

        store_knowledge(user_input, reply)

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})