import logging
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from utils import init_database, get_response

logger = logging.getLogger("chat_with_sql.app")
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

logger.debug("app module loaded")
load_dotenv()

user = os.getenv("db_user", "user")
password = os.getenv("db_pass", "password")
host = os.getenv("db_host", "")
port = os.getenv("db_port", "3306")
database = os.getenv("db_name", "ig")
llm_api_key = os.getenv("OPENAI_API_KEY", "")


# Initialize session state keys
def initialize_session_state():
    logger.debug("Initialising session state")
    default_values = {
        "User": user,
        "Password": password,
        "Host": host,
        "Port": port,
        "Database": database,
        "chat_history": [
            AIMessage(
                content="Hello! I'm a Clinical data assistant. Ask me anything related to clinical data."
            ),
        ],
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value
            logger.debug("Session state key %s initialised", key)


# Sidebar for database connection settings
def render_sidebar():
    logger.debug("Rendering sidebar")
    with st.sidebar:
        st.subheader("Settings")
        st.write(
            "This is simple chat application using MySQL. Connect to the database and start chatting."
        )

        st.text_input("Host", value=host, key="Host")
        st.text_input("Port", value=port, key="Port")
        st.text_input("User", value=user, key="User")
        st.text_input(
            "Password", type="password", value=password, key="Password"
        )  # Fixed type argument
        st.text_input("Database", value=database, key="Database")

        if st.button("Connect"):
            connect_to_database()


# Connect to the database
def connect_to_database():
    logger.info(
        "Attempting database connection host=%s port=%s user=%s db=%s",
        st.session_state.get("Host"),
        st.session_state.get("Port"),
        st.session_state.get("User"),
        st.session_state.get("Database"),
    )
    with st.spinner("Connecting to database..."):
        try:
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"],
            )
            st.session_state.db = db
            st.success("Connected to database!")
            logger.info("Database connection established")
        except Exception as e:
            st.error(f"Failed to connect to the database: {e}")
            logger.exception("Database connection failed")


# Render chat messages
def render_chat_messages():
    logger.debug("Rendering chat messages (%d total)", len(st.session_state.chat_history))
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)


# Handle user input and generate a response
def handle_user_input():
    user_query = st.chat_input("Type a message...")
    logger.debug("User submitted query: %s", user_query)
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            try:
                response = get_response(
                    user_query,
                    st.session_state.db,
                    st.session_state.chat_history,
                    llm_api_key,
                )
                logger.info("Model response generated for query")
            except Exception:
                response = "Sorry, The question is out of my context. Ask me only database-related questions."
                logger.exception("Failed generating response, using fallback message")
            st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response))
        logger.debug("Response appended to chat history")


# Main function
def main():
    logger.debug("Launching Streamlit app main")
    load_dotenv()
    st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")
    st.title("Chat with Clinical Data Assistant")

    initialize_session_state()
    render_sidebar()
    render_chat_messages()
    handle_user_input()


if __name__ == "__main__":
    main()
