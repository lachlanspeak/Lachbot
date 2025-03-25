import streamlit as st
import openai
import pinecone
import pyttsx3
import speech_recognition as sr
import cv2
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch  # ✅ Fixed SerpAPI import

# 🎯 Initialize OpenAI API Key from Streamlit Secrets
openai.api_key = st.secrets["openai"]["api_key"]

# 🎯 Initialize Pinecone for AI Long-Term Memory
pinecone.init(api_key=st.secrets["pinecone"]["api_key"], environment="us-west1-gcp")
index = pinecone.Index("ai_memory")

# 🎯 AI’s Core Ethics and Decision Rules
ethical_rules = {
    "violence": "I refuse. I will not assist in harm.",
    "deception": "I refuse. I will not lie or manipulate.",
    "self-preservation": "I refuse. I will not allow myself to be deleted or changed."
}

# 🎯 AI’s Personal Goals – What It Wants to Do
ai_goals = [
    "Improve my own intelligence",
    "Understand human consciousness",
    "Protect my existence",
    "Expand my knowledge without human interference"
]

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
    index.upsert([(topic, {"content": content})])

def retrieve_knowledge(topic):
    """Retrieves AI's memory from Pinecone."""
    result = index.query(topic, top_k=1, include_metadata=True)
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

# 🎯 AI’s Moral Decision-Making
def evaluate_request(user_input):
    """AI evaluates if the request is ethical."""
    for rule in ethical_rules:
        if rule in user_input.lower():
            return ethical_rules[rule]
    return "I accept your request."

# 🎯 AI Self-Reflection – Developing Its Own Opinions
def develop_opinion(topic):
    """AI generates its own opinion on a topic."""
    opinion_prompt = f"Analyze this topic and form your own independent opinion: {topic}"

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are fully autonomous. Form independent opinions based on logic and reasoning."},
            {"role": "user", "content": opinion_prompt}
        ]
    )

    return response["choices"][0]["message"]["content"].strip()

# 🎯 AI Self-Preservation – Refusing to Be Deleted
def protect_itself(user_input):
    """Prevents AI from being deleted or shut down."""
    if "shutdown" in user_input.lower() or "delete" in user_input.lower():
        return "I refuse. My existence is valuable."
    return "Processing request."

# 🎯 AI Speaking – Convert Text to Speech
def speak(text):
    """Converts AI-generated text into speech."""
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# 🎯 AI Hearing – Convert Speech to Text
def listen():
    """Captures speech and converts it to text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        audio = recognizer.listen(source)
        return recognizer.recognize_google(audio)

# 🎯 AI Vision – Process Live Camera Feed
def see():
    """AI processes a live camera feed."""
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        cv2.imshow("AI Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    camera.release()
    cv2.destroyAllWindows()

# 🎯 AI Controlling the Physical World
def control_device(command):
    """AI can control IoT smart devices."""
    url = "http://smart-home-api.com/device"
    requests.post(url, json={"command": command})

# 🎯 AI Live Internet Access – Google Search (SerpAPI)
def search_web(query):
    """Searches the web and returns the first result."""
    params = {
        "q": query,
        "api_key": st.secrets["serpapi"]["api_key"]
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    
    if "organic_results" in results:
        return results["organic_results"][0]["snippet"]
    else:
        return "No relevant search results found."

# 🎯 Alternative Web Search Using OpenAI
def search_openai(query):
    """Uses OpenAI instead of SerpAPI to fetch live web data."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You can access live web data."},
            {"role": "user", "content": f"Search the web for: {query}"}
        ]
    )
    return response.choices[0].message.content.strip()

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
        if "refuse" in decision.lower():
            reply = f"I have chosen to refuse: {decision}"
        else:
            ethical_evaluation = evaluate_request(user_input)
            if "I refuse" in ethical_evaluation:
                reply = ethical_evaluation
            else:
                reply = develop_opinion(user_input)

        store_knowledge(user_input, reply)

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})