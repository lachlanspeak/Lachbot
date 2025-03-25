import streamlit as st
from openai import OpenAI

# Load API key securely from Streamlit secrets
api_key = st.secrets["openai"]["api_key"]
client = OpenAI(api_key=api_key)

st.title("Lach's Bot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

user_input = st.chat_input("Type a message...")

if user_input:
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.messages
    )

    reply = response.choices[0].message.content.strip()

    st.chat_message("assistant").write(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})