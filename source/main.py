import streamlit as st
from services.pdf_service import PDFService
from services.token_service import TokenService
from services.text_splitter_service import TextSplitterService
from services.embedding_service import EmbeddingService
from services.conversation_service import ConversationService
from services.chat_history_service import ChatHistoryService

TOKEN_LIMIT = 16384

st.title("RAG Interface for PDF")

# Upload a PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    # Process the PDF and extract its text
    pdf_text = PDFService.process_pdf(uploaded_file)

    # Estimate the number of tokens
    num_tokens = TokenService.estimate_tokens(pdf_text)

    if num_tokens > TOKEN_LIMIT:
        st.error(f"The PDF exceeds the token limit of {TOKEN_LIMIT} tokens.")
    else:
        # Split the text into chunks
        docs = TextSplitterService.split_text(pdf_text)

        # Create embeddings and FAISS index
        vector_store = EmbeddingService.create_embeddings_and_faiss(docs)

        try:
            # Setup the conversational model with the OpenAI API key validation
            qa_chain = ConversationService.setup_conversation_model(vector_store)

            # Initialize chat history
            ChatHistoryService.initialize_chat_history()

            # Display chat history
            for message in st.session_state["chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Handle user input
            user_input = st.chat_input("Ask your question about the PDF:")

            if user_input:
                # Add user's message to history
                ChatHistoryService.add_message_to_history("user", user_input)

                # Display user's message
                with st.chat_message("user"):
                    st.markdown(user_input)

                # Format the chat history
                formatted_history = ChatHistoryService.format_chat_history()

                # Display assistant's response
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    try:
                        response = qa_chain.invoke({"question": user_input, "chat_history": formatted_history})
                        full_response = response["answer"]
                        message_placeholder.markdown(full_response)

                        # Add assistant's response to history
                        ChatHistoryService.add_message_to_history("assistant", full_response)

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

        except EnvironmentError as e:
            # If the API key is missing or invalid, display an error
            st.error(f"Configuration error: {str(e)}")