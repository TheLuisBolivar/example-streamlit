from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from configs.openai_config import OpenAIConfig  # Actualizado para usar OpenAIConfig

class ConversationService:
    @staticmethod
    def setup_conversation_model(vector_store):
        """Setup chat model and QA chain with OpenAI API key."""
        # Load the OpenAI API key from environment variables
        openai_api_key = OpenAIConfig.load_openai_api_key()
        
        # Initialize the chat model with the loaded API key
        chat_model = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", streaming=True)
        
        # Create the QA chain using the chat model and vector store
        qa_chain = ConversationalRetrievalChain.from_llm(chat_model, vector_store.as_retriever())
        
        return qa_chain