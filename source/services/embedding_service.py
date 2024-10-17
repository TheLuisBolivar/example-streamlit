from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from configs.openai_config import OpenAIConfig 

class EmbeddingService:
    @staticmethod
    def create_embeddings_and_faiss(docs):
        """Create embeddings and FAISS index from the documents."""
        openai_api_key = OpenAIConfig.load_openai_api_key()
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = FAISS.from_texts(docs, embeddings)
        return vector_store