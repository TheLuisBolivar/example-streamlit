import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter  # Esto queda igual
from langchain_community.vectorstores import FAISS  # Cambiado a langchain_community
from langchain_openai.embeddings import OpenAIEmbeddings  # Cambiado a langchain_openai
from langchain_openai.chat_models import ChatOpenAI  # Cambiado a langchain_openai
from langchain.chains import ConversationalRetrievalChain  # Esto queda en langchain
import tempfile
import os
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración de la API de OpenAI usando el archivo .env
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    st.error("Por favor, asegúrate de que la clave de OpenAI está definida en el archivo .env.")
else:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Límite de tokens del modelo
TOKEN_LIMIT = 16384

# Función para estimar el número de tokens (1 token ≈ 4 caracteres en inglés)
def estimate_tokens(text):
    return len(text) / 4

# Título de la aplicación
st.title("Interfaz RAG para PDF")

# Cargar el archivo PDF
uploaded_file = st.file_uploader("Sube un archivo PDF", type="pdf")

if uploaded_file is not None:
    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Leer el contenido del PDF usando PyPDF2
    reader = PdfReader(temp_path)
    pdf_text = ""
    for page in reader.pages:
        pdf_text += page.extract_text()

    # Estimar el número de tokens en el texto del PDF
    num_tokens = estimate_tokens(pdf_text)

    # Verificar si excede el límite de tokens
    if num_tokens > TOKEN_LIMIT:
        st.error(f"El PDF excede el límite de {TOKEN_LIMIT} tokens permitidos por el modelo. El archivo contiene aproximadamente {int(num_tokens)} tokens. Por favor, sube un archivo más pequeño.")
    else:
        # Dividir el texto en fragmentos manejables
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_text(pdf_text)

        # Crear las incrustaciones (embeddings) y el índice FAISS
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(docs, embeddings)

        # Crear el flujo de conversación con el modelo de chat
        chat_model = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)  # Habilitar el streaming en el modelo
        qa_chain = ConversationalRetrievalChain.from_llm(chat_model, vector_store.as_retriever())

        # Estado de la conversación
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # Mostrar el historial de chat
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Interfaz de chat usando chat_input
        user_input = st.chat_input("Escribe tu pregunta sobre el PDF:")

        if user_input:
            # Añadir el mensaje del usuario al historial
            st.session_state["chat_history"].append({"role": "user", "content": user_input})

            # Mostrar el mensaje del usuario
            with st.chat_message("user"):
                st.markdown(user_input)

            # Opción 2: Formatear el historial como una lista de tuplas
            formatted_history = [(msg["role"], msg["content"]) for msg in st.session_state["chat_history"]]

            # Mostrar el historial formateado para depuración
            st.write(formatted_history)  # Esto ayuda a ver cómo se formatea el historial antes de pasarlo

            # Mostrar el mensaje de respuesta mientras se genera (streaming)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    # Obtener respuesta del modelo RAG usando invoke() en lugar de run()
                    response = qa_chain.invoke({"question": user_input, "chat_history": formatted_history})

                    # Obtener la respuesta del modelo
                    full_response = response["answer"]  # Ajustar si la estructura de respuesta es diferente
                    message_placeholder.markdown(full_response)

                    # Guardar la respuesta en el historial
                    st.session_state["chat_history"].append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"Se produjo un error: {str(e)}")