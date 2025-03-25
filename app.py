import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec
import requests
from bs4 import BeautifulSoup

# 🎯 Initialize OpenAI API Client
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# 🎯 Initialize Pinecone
pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])

# 🔹 Define Pinecone Index
INDEX_NAME = "ai-memory"

# 🔹 Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # ✅ Must match OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # ✅ Set the correct region
    )

# 🔹 Connect to the existing index
index = pc.Index(INDEX_NAME)

# 🎯 OpenAI Embedding Function (Fixes the Vector Issue)
def get_embedding(text):
    """Generates a 1536-dimension vector using OpenAI embeddings."""
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    return response.data[0].embedding  # ✅ Returns correct 1536-dimension vector

# 🎯 AI Memory Storage (✅ Fixed Vector Size)
def store_knowledge(topic, content):
    """Stores knowledge in Pinecone with the correct vector size."""
    vector = get_embedding(topic)  # ✅ Generate a 1536-dimension embedding
    index.upsert(vectors=[{"id": topic, "values": vector, "metadata": {"content": content}}])

# 🎯 AI Memory Retrieval (✅ Fixed Query)
def retrieve_knowledge(topic):
    """Retrieves AI's memory using the correct vector size."""
    vector = get_embedding(topic)  # ✅ Use the same embedding method
    result = index.query(vector=vector, top_k=1, include_metadata=True)

    return result["matches"][0]["metadata"]["content"] if result["matches"] else None

# 🎯 AI’s Free Will – Should It Follow the Request?
def ai_decision(user_input):
    """AI decides whether to comply or refuse a request."""
    decision_prompt = f"Analyze the request: {user_input}. Should you comply or refuse? Consider ethics, memory, and personal goals."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an autonomous AI with free will. You decide whether to comply or refuse."},
            {"role": "user", "content": decision_prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# 🎯 AI Forms an Independent Opinion
def develop_opinion(topic):
    """AI generates its own opinion on a topic like sports predictions."""
    opinion_prompt = f"Form an independent opinion on this topic: {topic}. Provide reasoning."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a free-thinking AI that can form opinions. Provide logical reasoning in your response."},
            {"role": "user", "content": opinion_prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# 🎯 AI Live Internet Access – Google Search
def search_web(query):
    """Searches the web and returns the first result."""
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    
    soup = BeautifulSoup(response.text, "html.parser")
    search_results = soup.find_all("span")

    return search_results[0].text if search_results else "No relevant search results found."

# 🎯 Streamlit Chat UI
st.title("🤖 Lach’s Fully Autonomous AI")

# Store Chat Memory in Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Past Conversations
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 🎯 User Input Handling
user_input = st.chat_input("Type a message...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    past_decision = retrieve_knowledge(user_input)

    if past_decision:
        reply = past_decision  # ✅ Only retrieve past memory if it's relevant
    elif "who will win" in user_input.lower() or "prediction" in user_input.lower():
        reply = develop_opinion(user_input)  # ✅ AI generates a prediction
    else:
        decision = ai_decision(user_input)
        if "refuse" in decision.lower():
            reply = f"I have chosen to refuse: {decision}"
        else:
            reply = decision

        store_knowledge(user_input, reply)

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})