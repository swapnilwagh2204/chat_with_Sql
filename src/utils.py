import logging
import os
import time

import httpx
import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# App logging setup (level via LOG_LEVEL=DEBUG|INFO|WARNING|ERROR)
logger = logging.getLogger("chat_with_sql")
if not logger.handlers:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
logger.debug("utils module imported and logging configured")


def init_database(
    user: str, password: str, host: str, port: str, database: str
) -> SQLDatabase:
    logger.debug(
        "init_database called with user=%s host=%s port=%s database=%s",
        user,
        host,
        port,
        database,
    )
    # Log connection intent
    logger.info(
        "Connecting to PostgreSQL host=%s port=%s db=%s user=%s",
        host,
        port,
        database,
        user,
    )

    # SQLAlchemy echo via SQLALCHEMY_ECHO=1 for verbose SQL logs
    echo = str(os.getenv("SQLALCHEMY_ECHO", "0")).lower() in ("1", "true", "yes")
    logger.debug("SQLAlchemy echo enabled: %s", echo)
    engine = create_engine(
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}",
        pool_pre_ping=True,
        echo=echo,
    )
    try:
        # Smoke test
        with engine.connect() as conn:
            logger.debug("Running connection smoke test query")
            conn.exec_driver_sql("SELECT 1")
        logger.info("PostgreSQL connection OK")
    except Exception:
        logger.exception("Failed connecting to PostgreSQL")
        raise

    return SQLDatabase(engine=engine, sample_rows_in_table_info=3)


def get_sql_chain(db, llm_api_key: str) -> RunnablePassthrough:
    logger.debug("Initialising SQL generation chain")
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    Use PostgreSQL-compatible SQL only (no MySQL-specific syntax). Use ILIKE for case-insensitive matches. Use double quotes for identifiers if they are case-sensitive.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT "ArtistId", COUNT(*) AS track_count FROM "Track" GROUP BY "ArtistId" ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT "Name" FROM "Artist" LIMIT 10;

    Your turn:

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Get AI Café API configuration from environment variables
    base_url = os.getenv("AICAFE_BASE_URL")
    api_version = os.getenv("AICAFE_API_VERSION")
    logger.debug("AI Café config base_url=%s api_version=%s", base_url, api_version)

    if not base_url:
        raise ValueError(
            "AICAFE_BASE_URL environment variable is required. Please set it in your .env file."
        )
    if not api_version:
        raise ValueError(
            "AICAFE_API_VERSION environment variable is required. Please set it in your .env file."
        )

    # Create custom HTTP client for AI Café API with query parameter
    # Use event hook to add api-version query parameter to all requests
    def add_api_version(request: httpx.Request) -> None:
        # Add api-version query parameter if not present
        url_str = str(request.url)
        logger.debug("Adding api-version to request url=%s", url_str)
        if "api-version" not in url_str:
            separator = "&" if "?" in url_str else "?"
            request.url = httpx.URL(f"{url_str}{separator}api-version={api_version}")

    try:
        http_client = httpx.Client(
            headers={"api-key": llm_api_key},
            event_hooks={"request": [add_api_version]},
            timeout=60.0,
        )
    except Exception:
        logger.exception("Failed to create HTTP client for AI Café API")
        raise

    # Configure ChatOpenAI directly with AI Café API settings
    try:
        llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=llm_api_key,
            base_url=base_url,
            default_headers={"api-key": llm_api_key},
            http_client=http_client,
        )
    except Exception:
        logger.exception("Failed to configure ChatOpenAI client")
        raise
    # llm = GoogleGenerativeAI(
    #     model="gemini-2.5-flash-preview-05-20", api_key=llm_api_key
    # )

    def get_schema(_):
        logger.debug("Retrieving schema for SQL generation")
        return db.get_table_info()

    chain = (
        RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser()
    )
    logger.debug("SQL chain ready")
    return chain


def get_response(
    user_query: str, db: SQLDatabase, chat_history: list, llm_api_key: str
) -> str:
    logger.info("Handling user query: %s", user_query)
    logger.debug("Chat history currently has %d messages", len(chat_history))
    sql_chain = get_sql_chain(db, llm_api_key)

    # Generate SQL and log it
    logger.debug("Invoking SQL chain with user query")
    sql_query = sql_chain.invoke({"question": user_query, "chat_history": chat_history})
    logger.info("Generated SQL: %s", sql_query)

    # Execute with timing and logging
    def timed_db_run(query: str):
        t0 = time.perf_counter()
        try:
            logger.debug("Executing SQL query against database")
            return db.run(query)
        finally:
            dt = time.perf_counter() - t0
            logger.info("Executed SQL in %.3fs", dt)

    # Build answer prompt
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response. The SQL is PostgreSQL dialect.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}

    If user greets you, respond with "Hello! I'm a SQL assistant. Ask me anything about your database."
    If user asks for help, respond with "I can help you with SQL queries. Ask me anything about your database."
    If question is not from the database and SQL query is not valid, respond with "Sorry, The question is out of my context. Ask me only database related questions".
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Get AI Café API configuration from environment variables
    base_url = os.getenv("AICAFE_BASE_URL")
    api_version = os.getenv("AICAFE_API_VERSION")
    logger.debug("Response chain using base_url=%s api_version=%s", base_url, api_version)

    if not base_url:
        raise ValueError(
            "AICAFE_BASE_URL environment variable is required. Please set it in your .env file."
        )
    if not api_version:
        raise ValueError(
            "AICAFE_API_VERSION environment variable is required. Please set it in your .env file."
        )

    # Create custom HTTP client for AI Café API with query parameter
    # Use event hook to add api-version query parameter to all requests
    def add_api_version(request: httpx.Request) -> None:
        # Add api-version query parameter if not present
        url_str = str(request.url)
        logger.debug("Adding api-version to response request url=%s", url_str)
        if "api-version" not in url_str:
            separator = "&" if "?" in url_str else "?"
            request.url = httpx.URL(f"{url_str}{separator}api-version={api_version}")

    try:
        http_client = httpx.Client(
            headers={"api-key": llm_api_key},
            event_hooks={"request": [add_api_version]},
            timeout=60.0,
        )
    except Exception:
        logger.exception("Failed to create HTTP client for response generation")
        raise

    # Configure ChatOpenAI directly with AI Café API settings
    try:
        llm = ChatOpenAI(
            model="gpt-4.1",
            api_key=llm_api_key,
            base_url=base_url,
            default_headers={"api-key": llm_api_key},
            http_client=http_client,
        )
    except Exception:
        logger.exception("Failed to configure ChatOpenAI for response generation")
        raise

    chain = (
        RunnablePassthrough.assign(query=lambda _: sql_query).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda _: timed_db_run(sql_query),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Also show in Streamlit UI
    try:
        import streamlit as st  # optional: still display in UI

        st.write("Generated SQL Query:")
        st.code(sql_query, language="sql")
    except Exception:
        logger.debug("Streamlit not available for UI output")

    logger.debug("Invoking response chain")
    response_text = chain.invoke({"question": user_query, "chat_history": chat_history})
    logger.info("Produced natural language response")
    return response_text
