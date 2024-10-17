import unittest
import streamlit as st
from services.chat_history_service import ChatHistoryService

class TestChatHistoryService(unittest.TestCase):

    def test_initialize_chat_history(self):
        ChatHistoryService.initialize_chat_history()
        self.assertIn("chat_history", st.session_state)

    def test_add_message_to_history(self):
        ChatHistoryService.initialize_chat_history()
        ChatHistoryService.add_message_to_history("user", "Hello")
        self.assertEqual(len(st.session_state["chat_history"]), 1)
        self.assertEqual(st.session_state["chat_history"][0]["content"], "Hello")

    def test_format_chat_history(self):
        ChatHistoryService.initialize_chat_history()
        ChatHistoryService.add_message_to_history("user", "Hello")
        history = ChatHistoryService.format_chat_history()
        self.assertEqual(history[0][1], "Hello")

if __name__ == '__main__':
    unittest.main()