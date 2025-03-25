import streamlit as st
import openai
import pinecone
import os

# 🔑 Load API Keys from secrets.toml
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
PINECONE_ENV = st.secrets["pinecone"]["environment"]

# ✅ Initialize OpenAI
openai.api_key = OPENAI_API_KEY

# ✅ Initialize Pinecone
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "ai-memory"

# 🚀 Ensure the index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # ✅ Adjust if needed
        metric="cosine"
    )

# 🔄 Connect to the Pinecone index
index = pc.Index(INDEX_NAME)

# 🎯 Streamlit Chat UI
st.title("🤖 Lach’s Fully Autonomous AI")

# Store Chat Memory in Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Past Conversations
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# ✅ Fix 1: Prevent KeyError by checking metadata
def retrieve_knowledge(topic):
    """Retrieves AI's memory from Pinecone safely."""
    vector = get_embedding(topic)
    result = index.query(vector=vector, top_k=1, include_metadata=True)

    if result["matches"]:
        metadata = result["matches"][0].get("metadata", {})
        return metadata.get("response", None)  # ✅ Check if "response" exists

    return None  # No matching knowledge found

# ✅ Fix 2: Ensure knowledge is stored properly
def store_knowledge(topic, response):
    """Stores knowledge into Pinecone memory with a valid embedding."""
    vector = get_embedding(topic)
    index.upsert(vectors=[{
        "id": topic,
        "values": vector,
        "metadata": {"response": response}  # ✅ Ensure "response" is stored
    }])

# ✅ Fix 3: Debugging Output if Needed
def debug_knowledge(topic):
    """Debugs what is stored in Pinecone."""
    vector = get_embedding(topic)
    result = index.query(vector=vector, top_k=1, include_metadata=True)
    st.write(f"🔍 Debugging Pinecone Data: {result}")

# ✅ Ensure OpenAI embedding works
def get_embedding(text):
    """Generates an embedding for text using OpenAI."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# 🎯 AI Decision Logic
def ai_decision(user_input):
    """AI decides whether to comply or refuse a request."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"Analyze: {user_input}"}]
    )
    return response.choices[0].message.content.strip()

# 🎯 Handle User Input
user_input = st.chat_input("Type a message...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    past_response = retrieve_knowledge(user_input)

    if past_response:
        reply = f"🤖 (Memory Recall): {past_response}"
    else:
        reply = ai_decision(user_input)
        store_knowledge(user_input, reply)

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

    # 🔍 Debug if issues persist
    debug_knowledge(user_input)


