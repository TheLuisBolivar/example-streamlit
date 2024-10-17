import unittest
from unittest.mock import patch
from services.embedding_service import EmbeddingService

class TestEmbeddingService(unittest.TestCase):

    @patch('services.embedding_service.OpenAIEmbeddings')
    @patch('services.embedding_service.FAISS')
    def test_create_embeddings_and_faiss(self, mock_faiss, mock_embeddings):
        docs = ["This is a test.", "Another document."]
        
        mock_embeddings_instance = mock_embeddings.return_value
        
        mock_faiss.return_value
        
        EmbeddingService.create_embeddings_and_faiss(docs)
        
        mock_faiss.from_texts.assert_called_once_with(docs, mock_embeddings_instance)

if __name__ == '__main__':
    unittest.main()