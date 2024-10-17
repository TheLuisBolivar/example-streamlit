import streamlit as st

class ChatHistoryService:
    @staticmethod
    def initialize_chat_history():
        """Initialize chat history in session state."""
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

    @staticmethod
    def add_message_to_history(role, content):
        """Add a message to the chat history."""
        st.session_state["chat_history"].append({"role": role, "content": content})

    @staticmethod
    def format_chat_history():
        """Format chat history as a list of tuples."""
        return [(msg["role"], msg["content"]) for msg in st.session_state["chat_history"]]