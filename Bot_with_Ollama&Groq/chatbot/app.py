import streamlit as st
import os
from dotenv import load_dotenv
from groq import groqclient as GroqClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# Set environment variables for Groq API
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Initialize Groq client
groq_client = GroqClient(api_key=os.getenv("GROQ_API_KEY"))

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question: {question}")
    ]
)

# Streamlit framework
st.title("Langchain Demo with Groq API")
input_text = st.text_input("Search the topic you want")

# Groq LLM invocation function
def invoke_groq_chain(prompt, question):
    chain = prompt | groq_client | StrOutputParser()
    response = chain.invoke({'question': question})
    return response

if input_text:
    response = invoke_groq_chain(prompt, input_text)
    st.write(response)
