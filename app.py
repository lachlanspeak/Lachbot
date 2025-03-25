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

# ✅ Fix: Prevent storing junk data
def is_valid_input(text):
    """Checks if input is valid for storing memory."""
    return text.strip() and len(text.split()) > 1

# ✅ Fix: Retrieve past knowledge correctly
def retrieve_knowledge(topic):
    """Retrieves AI's memory from Pinecone safely."""
    vector = get_embedding(topic)
    result = index.query(vector=vector, top_k=1, include_metadata=True)

    if result["matches"]:
        metadata = result["matches"][0].get("metadata", {})
        response = metadata.get("response")
        return response if response else None

    return None  # No matching memory found

# ✅ Fix: Store knowledge properly
def store_knowledge(topic, response):
    """Stores knowledge into Pinecone memory correctly."""
    if is_valid_input(response):  # ✅ Prevent storing "Hello" or meaningless text
        vector = get_embedding(topic)
        index.upsert(vectors=[{
            "id": topic,
            "values": vector,
            "metadata": {"response": response}
        }])

# ✅ Fix: Ensures OpenAI embedding works
def get_embedding(text):
    """Generates an embedding for text using OpenAI."""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

# ✅ Fix: More human-like AI decision logic
def ai_decision(user_input):
    """AI generates intelligent responses based on context."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_input}]
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



