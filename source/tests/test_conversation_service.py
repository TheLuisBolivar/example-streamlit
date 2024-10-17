import unittest
from unittest.mock import patch, MagicMock
from services.conversation_service import ConversationService

class TestConversationService(unittest.TestCase):

    @patch('services.conversation_service.ChatOpenAI')
    @patch('services.conversation_service.ConversationalRetrievalChain')
    def test_setup_conversation_model(self, mock_chain, mock_chat):
        # Crear un mock para el vector_store
        mock_vector_store = MagicMock()
        mock_vector_store.as_retriever.return_value = "mock retriever"

        mock_chat_instance = mock_chat.return_value

        mock_chain_instance = mock_chain.from_llm.return_value

        result = ConversationService.setup_conversation_model(mock_vector_store)

        mock_chain.from_llm.assert_called_once_with(mock_chat_instance, "mock retriever")
        self.assertEqual(result, mock_chain_instance)

if __name__ == '__main__':
    unittest.main()