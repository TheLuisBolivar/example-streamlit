from langchain.text_splitter import CharacterTextSplitter

class TextSplitterService:
    @staticmethod
    def split_text(text, chunk_size=1000, chunk_overlap=200):
        """Split the text into manageable chunks."""
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_text(text)