import os
from dotenv import load_dotenv
import streamlit as st

class OpenAIConfig:
    @staticmethod
    def load_openai_api_key():
        """Load and validate the OpenAI API key from the environment."""
        load_dotenv()  # Load environment variables from .env file
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            # Display error in the Streamlit app and raise an exception
            st.error("Please ensure that the OpenAI API key is set in the .env file.")
            raise EnvironmentError("The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable.")
        
        return openai_api_key