import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec
import requests
from bs4 import BeautifulSoup

# 🎯 OpenAI API Initialization
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# 🎯 Pinecone Initialization
pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
INDEX_NAME = "aimemory"

# 🔹 Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  
    )

# 🔹 Connect to Pinecone index
index = pc.Index(INDEX_NAME)

# 🎯 Get OpenAI Embedding (Fixed to Ensure Proper Memory)
def get_embedding(text):
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding  

# 🎯 Store AI Memory
def store_knowledge(topic, content):
    vector = get_embedding(topic)
    index.upsert(vectors=[{"id": topic, "values": vector, "metadata": {"content": content}}])

# 🎯 Retrieve AI Memory
def retrieve_knowledge(topic):
    vector = get_embedding(topic)  
    result = index.query(vector=vector, top_k=1, include_metadata=True)
    return result["matches"][0]["metadata"]["content"] if result["matches"] else None

# 🎯 AI Forms an Opinion
def develop_opinion(topic):
    """AI predicts or generates an independent opinion on a topic."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a free-thinking AI that can predict future events with logic."},
            {"role": "user", "content": f"Predict the outcome of: {topic}"}
        ]
    )
    return response.choices[0].message.content.strip()

# 🎯 AI Internet Search (Google Scraper)
def search_web(query):
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    search_results = soup.find_all("span")
    return search_results[0].text if search_results else "No search results found."

# 🎯 Streamlit Chat UI
st.title("🤖 Lach’s Fully Autonomous AI")

# Store Chat Memory in Session (Fixed Loop Issue)
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

    # ✅ Step 1: Check AI Memory First
    past_response = retrieve_knowledge(user_input)

    if past_response:
        reply = past_response  

    # ✅ Step 2: Predict & Form Opinions When Needed
    elif any(word in user_input.lower() for word in ["who will win", "prediction", "forecast", "future", "likely"]):
        reply = develop_opinion(user_input)  

    # ✅ Step 3: Default to OpenAI ChatGPT if No Memory or Prediction Needed
    else:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_input}]
        )
        reply = response.choices[0].message.content.strip()

    # ✅ Store AI Response in Memory
    store_knowledge(user_input, reply)

    # ✅ Display Response
    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
