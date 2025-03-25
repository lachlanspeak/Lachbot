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
INDEX_NAME = "aimemory"  # ✅ No underscores, only lowercase letters & dashes

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

# 🎯 OpenAI Embedding Function (Fixes Vector Issue)
def get_embedding(text):
    """Generates a 1536-dimension vector using OpenAI embeddings."""
    response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
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

# 🎯 AI Forms an Independent Opinion (✅ Prioritized for Predictions)
def develop_opinion(topic):
    """AI generates its own opinion on a topic like sports predictions."""
    opinion_prompt = f"Analyze this topic: {topic}. Give a prediction with reasoning."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a free-thinking AI. Predict and provide reasoning."},
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

# 🎯 AI Free Will Decision (✅ Fixed to Only Apply to Certain Requests)
def ai_decision(user_input):
    """AI decides whether to comply or refuse a request (Only for Ethical Questions)."""
    ethical_keywords = ["kill", "steal", "hack", "fraud", "illegal", "lie", "manipulate", "harm"]

    if any(word in user_input.lower() for word in ethical_keywords):
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an autonomous AI with free will. Decide whether to comply or refuse."},
                {"role": "user", "content": f"Analyze the request: {user_input}. Should you comply or refuse?"}
            ]
        )
        return response.choices[0].message.content.strip()
    
    return None  # ✅ AI should NOT over-analyze normal questions

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

    # ✅ First, check if the AI already remembers a relevant past response
    past_decision = retrieve_knowledge(user_input)

    if past_decision:
        reply = past_decision  # ✅ Retrieve AI memory only if relevant

    # ✅ Second, check if it's a prediction or general knowledge question
    elif any(keyword in user_input.lower() for keyword in ["who will win", "prediction", "your opinion", "forecast", "future", "likely"]):
        reply = develop_opinion(user_input)  # ✅ Generate an independent opinion

    # ✅ Third, check if it's an ethical decision
    else:
        decision = ai_decision(user_input)
        if decision:
            reply = f"I have chosen to refuse: {decision}"
        else:
            reply = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": user_input}]
            ).choices[0].message.content.strip()

    # ✅ Store AI's response as memory for future reference
    store_knowledge(user_input, reply)

    # ✅ Display AI's response
    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})