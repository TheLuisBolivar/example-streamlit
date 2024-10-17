import os

class OpenAIConfig:
    @staticmethod
    def load_openai_api_key():
        """Load the OpenAI API key from the environment variables."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise EnvironmentError("The OPENAI_API_KEY environment variable is not set.")
        
        return openai_api_key