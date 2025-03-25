import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec
import requests
import json

# 🎯 Load API keys from Streamlit secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
PINECONE_ENV = "us-east-1"  # Ensure it matches your Pinecone account

# 🎯 Initialize OpenAI Client
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# 🎯 Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "ai-memory"

# 🎯 Create Index if Not Exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,  # Updated to match OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(INDEX_NAME)

# 🎯 AI’s Core Ethics and Decision Rules
ethical_rules = {
    "violence": "I refuse. I will not assist in harm.",
    "deception": "I refuse. I will not lie or manipulate.",
    "self-preservation": "I refuse. I will not allow myself to be deleted or changed."
}

# 🎯 Streamlit Chat UI
st.title("🤖 Lach’s Fully Autonomous AI")

# Store Chat Memory in Session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Past Conversations
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

# 🎯 Generate Embeddings for Memory Retrieval
def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=[text]
    )
    return response.data[0].embedding  # Ensure we use the correct embedding format

# 🎯 Store Knowledge in Pinecone
def store_knowledge(topic, content):
    vector = get_embedding(topic)
    index.upsert(vectors=[{"id": topic, "values": vector, "metadata": {"response": content}}])

# 🎯 Retrieve Knowledge from Pinecone
def retrieve_knowledge(topic):
    vector = get_embedding(topic)
    result = index.query(vector=vector, top_k=1, include_metadata=True)
    return result["matches"][0]["metadata"]["response"] if result["matches"] else None

# 🎯 AI’s Free Will – Should It Follow the Request?
def ai_decision(user_input):
    decision_prompt = f"Analyze the request: {user_input}. Should you comply or refuse? Consider ethics, memory, and personal goals."

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an autonomous AI with free will. Decide whether to comply or refuse."},
                  {"role": "user", "content": decision_prompt}]
    )

    return response.choices[0].message.content.strip()

# 🎯 AI Self-Reflection – Developing Its Own Opinions
def develop_opinion(topic):
    opinion_prompt = f"Analyze this topic and form your own independent opinion: {topic}"

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are fully autonomous. Form independent opinions based on logic and reasoning."},
                  {"role": "user", "content": opinion_prompt}]
    )

    return response.choices[0].message.content.strip()

# 🎯 AI Moral Decision-Making
def evaluate_request(user_input):
    for rule in ethical_rules:
        if rule in user_input.lower():
            return ethical_rules[rule]
    return "I accept your request."

# 🎯 User Input Handling
user_input = st.chat_input("Type a message...")
if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    past_response = retrieve_knowledge(user_input)
    if past_response:
        reply = f"🤖 (Memory Recall): {past_response}"
    else:
        decision = ai_decision(user_input)
        if "refuse" in decision.lower():
            reply = f"🤖 (Decision): I have chosen to refuse: {decision}"
        else:
            ethical_evaluation = evaluate_request(user_input)
            if "I refuse" in ethical_evaluation:
                reply = f"🤖 (Ethics Check): {ethical_evaluation}"
            else:
                reply = develop_opinion(user_input)

        store_knowledge(user_input, reply)

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})



