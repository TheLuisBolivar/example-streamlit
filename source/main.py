import streamlit as st
from services.pdf_service import PDFService
from services.token_service import TokenService
from services.text_splitter_service import TextSplitterService
from services.embedding_service import EmbeddingService
from services.conversation_service import ConversationService
from services.chat_history_service import ChatHistoryService
from utils.constants import AppConstants

st.title("RAG Interface for PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    pdf_text = PDFService.process_pdf(uploaded_file)
    num_tokens = TokenService.estimate_tokens(pdf_text)

    if num_tokens > AppConstants.TOKEN_LIMIT:
        st.error(f"The PDF exceeds the token limit of {AppConstants.TOKEN_LIMIT} tokens.")
    else:
        docs = TextSplitterService.split_text(pdf_text)
        vector_store = EmbeddingService.create_embeddings_and_faiss(docs)

        try:
            qa_chain = ConversationService.setup_conversation_model(vector_store)
            ChatHistoryService.initialize_chat_history()

            for message in st.session_state["chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            user_input = st.chat_input("Ask your question about the PDF:")

            if user_input:
                ChatHistoryService.add_message_to_history("user", user_input)

                with st.chat_message("user"):
                    st.markdown(user_input)

                formatted_history = ChatHistoryService.format_chat_history()

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    try:
                        response = qa_chain.invoke({"question": user_input, "chat_history": formatted_history})
                        full_response = response["answer"]
                        message_placeholder.markdown(full_response)

                        ChatHistoryService.add_message_to_history("assistant", full_response)

                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

        except EnvironmentError as e:
            st.error(f"Configuration error: {str(e)}")