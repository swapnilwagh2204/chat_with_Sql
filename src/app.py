from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from utils import init_database, get_response


# Initialize session state keys
def initialize_session_state():
    default_values = {
        "User": "root",
        "Password": "",
        "Host": "localhost",
        "Port": "3306",
        "Database": "ig",
        "chat_history": [
            AIMessage(
                content="Hello! I'm a SQL assistant. Ask me anything about your database."
            )
        ],
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


# Sidebar for database connection settings
def render_sidebar():
    with st.sidebar:
        st.subheader("Settings")
        st.write(
            "This is a simple chat application using MySQL. Connect to the database and start chatting."
        )

        st.text_input("Host", value="localhost", key="Host")
        st.text_input("Port", value="3306", key="Port")
        st.text_input("User", value="root", key="User")
        st.text_input("Password", type="password", value="", key="Password")
        st.text_input("Database", value="ig", key="Database")

        if st.button("Connect"):
            connect_to_database()


# Connect to the database
def connect_to_database():
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
        except Exception as e:
            st.error(f"Failed to connect to the database: {e}")


# Render chat messages
def render_chat_messages():
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
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            try:
                response = get_response(
                    user_query, st.session_state.db, st.session_state.chat_history
                )
            except Exception:
                response = "Sorry, The question is out of my context. Ask me only database-related questions."
            st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response))


# Main function
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")
    st.title("Chat with MySQL Database")

    initialize_session_state()
    render_sidebar()
    render_chat_messages()
    handle_user_input()


if __name__ == "__main__":
    main()
