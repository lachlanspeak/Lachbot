import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec
import requests
from bs4 import BeautifulSoup

# 🎯 OpenAI API Key
client = openai.OpenAI(api_key=st.secrets["openai"]["api_key"])

# 🎯 Pinecone Initialization
pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
INDEX_NAME = "aimemory"

# 🔹 Ensure the Pinecone index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  
    )

index = pc.Index(INDEX_NAME)

# 🎯 Get OpenAI Embedding
def get_embedding(text):
    """Convert text into a vector for Pinecone storage."""
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
    return response.data[0].embedding  

# 🎯 Store AI Knowledge (Fixed)
def store_knowledge(user_input, response_text):
    """Store user query + AI response in Pinecone memory."""
    vector = get_embedding(user_input)
    index.upsert(vectors=[{"id": user_input, "values": vector, "metadata": {"query": user_input, "response": response_text}}])

# 🎯 Retrieve AI Memory (Fixed)
def retrieve_knowledge(user_input):
    """Retrieve stored AI response from Pinecone memory."""
    vector = get_embedding(user_input)  
    result = index.query(vector=vector, top_k=1, include_metadata=True)
    return result.matches[0].metadata["response"] if result.matches else None

# 🎯 AI Predicts Future Events
def develop_opinion(topic):
    """AI forms an opinion or prediction about an event."""
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an AI that predicts future events based on logic and trends."},
            {"role": "user", "content": f"Predict the outcome of: {topic}"}
        ]
    )
    return response.choices[0].message.content.strip()

# 🎯 AI Internet Search
def search_web(query):
    """Searches the web and returns the first relevant result."""
    search_url = f"https://www.google.com/search?q={query}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    search_results = soup.find_all("span")
    return search_results[0].text if search_results else "No search results found."

# 🎯 Streamlit Chat UI
st.title("🤖 Lach’s Fully Autonomous AI")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Previous Conversations
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 🎯 User Input Handling (Fixed)
user_input = st.chat_input("Type a message...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ✅ Step 1: Retrieve Memory First
    past_response = retrieve_knowledge(user_input)

    if past_response:
        reply = f"🤖 (Memory Recall): {past_response}"

    # ✅ Step 2: Predict & Form Opinions If Needed
    elif any(word in user_input.lower() for word in ["who will win", "prediction", "forecast", "future", "likely"]):
        reply = develop_opinion(user_input)  

    # ✅ Step 3: Perform Web Search If Needed
    elif "search" in user_input.lower():
        reply = search_web(user_input.replace("search", "").strip())

    # ✅ Step 4: Default to OpenAI ChatGPT for Everything Else
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

