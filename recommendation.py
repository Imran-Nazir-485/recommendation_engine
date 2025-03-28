import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
import os
import torch
import ast
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import sqlite3
import json
import numpy as np
import gdown
from dotenv import load_dotenv
import random
import re
import faiss
from openai import OpenAI
from langchain.prompts import PromptTemplate
# Load environment variables
load_dotenv()
# Load Sentence Transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')
# Page Layout
# Load an open-source embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

import pickle
import faiss

# # https://drive.google.com/file/d/1qkcNnBR9m7ivbx69HawYoe-QLO2SewYp/view?usp=sharing
# output_file = "faiss_index.bin"
# # Download the file
# @st.cache_data
# def download_db():
#     url = f"https://drive.google.com/uc?id=1qkcNnBR9m7ivbx69HawYoe-QLO2SewYp"
#     gdown.download(url, output_file, quiet=False)
#     return output_file
# f=download_db()

# # Load FAISS index
# index = faiss.read_index(f)
# # https://drive.google.com/file/d/1wYz0VHandcxXrr3krT7ILdtFYeYTL05v/view?usp=sharing
# # Download the file
# output_file = "metadata.pkl"
# @st.cache_data
# def download_db():
#     url = f"https://drive.google.com/uc?id=1wYz0VHandcxXrr3krT7ILdtFYeYTL05v"
#     gdown.download(url, output_file, quiet=False)
#     return output_file
    
# m=download_db()
# # Load metadata
# with open(m, "rb") as mk:
#     metadata = pickle.load(mk)


# # Load FAISS index before searching
# def search_places(user_profile_text, top_k=5):
#     """Search for places similar to a user profile."""
    
#     # Load FAISS index
#     # index = faiss.read_index(f)

#     # Generate embedding for user profile
#     user_embedding = model.encode([user_profile_text], convert_to_numpy=True)

#     # Perform similarity search
#     distances, indices = index.search(user_embedding, top_k)

#     # Load metadata
#     # with open(m, "rb") as f:
#     #     metadata = pickle.load(f)

#     # Retrieve search results
#     results = [metadata[i] for i in indices[0]]
    
#     return results


GROQ_API_KEY=os.getenv("GROQ_API_KEY")

from langchain_groq import ChatGroq

llm = ChatGroq(
    temperature=0,
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY
)


# Define a custom prompt template
def custom_PromptTemplate(context):
    return f"""
    You are an AI assistant for sentiment and emotion analysis.  
    
    Context:  
    {context}  
    
    Provide the analysis results.
    Provide Label.
    Small Analysis Report
    Represent results beatifully and use appropriate emojis.
    """





# file_id = "1ug8pf1M1tes-CJMhS_sso372tvC4RQv8"
# output_file = "open_ai_key.txt"

# # https://docs.google.com/spreadsheets/d/1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g/edit?gid=0#gid=0
# sheet_id = '1Dp6Y9ps4md393F5eRZzaZhu044k4JCmrbYDxWmQ6t2g' # replace with your sheet's ID
# url=f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
# df=pd.read_csv(url)
# os.environ["OPENAI_API_KEY"] = df.keys()[0]





# Set the page configuration to wide mode
# st.set_page_config(page_title="My App", layout="wide")



st.subheader("Understand Text Emotion with AI")


user_profile=st.text_area("Add Text Here")

if st.button("Analyze"):
    # Example user profile search
    # user_profile = "I love historical sites and museums with cultural exhibits. I prefer places that are quiet and educational."
    # results = search_places(user_profile)
    # res=""
    # Print search results
    # for res in results:
    #     res= " " + f"Place: Category: {res['Category']}, City: {res['City']}, Country: {res['Country']} Tags: {res['Tags']}, Best time to visit: {res['Best Time to Visit']}, Additional Info: {res['Additional Info']}"
        # st.write(f"Place: Category: {res['Category']}, City: {res['City']}, Country: {res['Country']} Tags: {res['Tags']}, Best time to visit: {res['Best Time to Visit']}, Additional Info: {res['Additional Info']}")
    st.write(llm.invoke(custom_PromptTemplate(user_profile)).content)


