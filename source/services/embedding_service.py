from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from configs.openai_config import OpenAIConfig  # Cargar configuración de OpenAI

class EmbeddingService:
    @staticmethod
    def create_embeddings_and_faiss(docs):
        """Create embeddings and FAISS index from the documents."""
        # Load the OpenAI API key from environment variables
        openai_api_key = OpenAIConfig.load_openai_api_key()
        
        # Create OpenAIEmbeddings using the API key
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Create FAISS index from the document embeddings
        vector_store = FAISS.from_texts(docs, embeddings)
        return vector_store