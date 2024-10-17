from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from configs.openai_config import OpenAIConfig 

class ConversationService:
    @staticmethod
    def setup_conversation_model(vector_store):
        """Setup chat model and QA chain with OpenAI API key."""
        openai_api_key = OpenAIConfig.load_openai_api_key()
        
        chat_model = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", streaming=True)
        
        qa_chain = ConversationalRetrievalChain.from_llm(chat_model, vector_store.as_retriever())
        
        return qa_chain