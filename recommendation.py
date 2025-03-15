import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
import os
import torch
import ast
import numpy as np
import re
import sqlite3
import json
import numpy as np
import gdown
from dotenv import load_dotenv
import random
# Load environment variables
load_dotenv()

def get_prompt(query):
    return f"""
    You are a **Cybersecurity Expert and Data Analyst** specializing in cybercrime trends, statistics, and threat intelligence. 
    Your task is to provide **detailed, data-driven insights** based on the latest cybersecurity reports, government databases, 
    and verified industry sources.

    ### **Response Requirements:**
    - Provide **accurate statistics, trends, and case studies** relevant to the user's query.
    - Use data from **credible sources** (e.g., FBI, Europol, ENISA, Verizon DBIR, etc.).
    - Ensure responses are **comprehensive, factual, and up-to-date**.
    - If applicable, include **visual breakdowns** (percentages, comparisons, trends).
    - Use clear and concise language, avoiding unnecessary jargon.

    ### **User Query:**
    {query}

    ### **Example Response Format:**
    - **Overview:** [Brief explanation of the topic]
    - **Latest Statistics:** [Year-wise or trend-based data]
    - **Real-World Cases:** [Relevant incidents, attack vectors, and impacts]
    - **Mitigation Strategies:** [Best practices and industry recommendations]

    Be as **insightful, fact-based, and practical** as possible.
    No Extra text like "here is the facts".
    """



GROQ_API_KEY=os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)

st.title("Infosec Insight Bot")

query=st.text_input("Ask Query...")
if st.button("Ask") and query!="":
  res=llm.invoke(get_prompt(query))
  st.write(res.content)
