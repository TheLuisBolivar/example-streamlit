import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import tempfile
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    # Check if OpenAI API key is missing
    st.error("Please ensure that the OpenAI key is defined in the .env file.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

TOKEN_LIMIT = 16384

def estimate_tokens(text):
    return len(text) / 4

st.title("RAG Interface for PDF")

# Check if a PDF file has been uploaded
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded PDF file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    reader = PdfReader(temp_path)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()

    num_tokens = estimate_tokens(pdf_text)

    # Check if the PDF exceeds the token limit
    if num_tokens > TOKEN_LIMIT:
        st.error(f"The PDF exceeds the token limit of {TOKEN_LIMIT} tokens. The file contains approximately {int(num_tokens)} tokens. Please upload a smaller file.")
    else:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_text(pdf_text)

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(docs, embeddings)

        chat_model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(chat_model, vector_store.as_retriever())

        # Check if chat history is initialized in session state
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Check if user input is provided
        user_input = st.chat_input("Ask your question about the PDF:")

        if user_input:
            # Add user input to chat history
            st.session_state["chat_history"].append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            formatted_history = [(msg["role"], msg["content"]) for msg in st.session_state["chat_history"]]

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    # Invoke the conversational retrieval chain to get a response
                    response = qa_chain.invoke({"question": user_input, "chat_history": formatted_history})
                    full_response = response["answer"]
                    message_placeholder.markdown(full_response)

                    # Add assistant's response to chat history
                    st.session_state["chat_history"].append({"role": "assistant", "content": full_response})

                except Exception as e:
                    # Display any error that occurs
                    st.error(f"An error occurred: {str(e)}")