import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


def init_database(
    user: str, password: str, host: str, port: str, database: str
) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)


def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI(model="gpt-4-0125-preview")

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema) | prompt | llm | StrOutputParser()
    )


def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)

    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
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

    llm = ChatOpenAI(model="gpt-4-0125-preview")

    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate the SQL query
    sql_query = sql_chain.invoke(
        {
            "question": user_query,
            "chat_history": chat_history,
        }
    )

    # Print the SQL query to the Streamlit app
    st.write("Generated SQL Query:")
    st.code(sql_query, language="sql")

    return chain.invoke(
        {
            "question": user_query,
            "chat_history": chat_history,
        }
    )
