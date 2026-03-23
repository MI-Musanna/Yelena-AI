import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma 
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- UI SETUP ---
st.set_page_config(page_title="Yelena AI", page_icon="⚡")

# --- AUTO-BRAIN LOGIC (The "No-Manual" Part) ---
@st.cache_resource
def initialize_yelena():
    my_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = my_key
    
    # 1. AUTO-SCRAPE: Tell her where to look
    urls = ["https://mitecharsenal.blogspot.com/"] 
    loader = WebBaseLoader(urls)
    data = loader.load()
    
    # 2. AUTO-CHUNK: Break it down
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    
    # 3. AUTO-EMBED: Build the brain in the cloud
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # We use an in-memory database so we don't need to upload a folder!
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    
    # 4. SETUP LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=my_key)
    
    system_prompt = "You are Yelena, Mahdi's tech assistant. Use this context: {context}"
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    
    return ({"context": vectorstore.as_retriever(search_kwargs={"k": 5}), "question": RunnablePassthrough()} 
            | prompt | llm | StrOutputParser())

chain = initialize_yelena()

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if user_input := st.chat_input("Ask me about the blog..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"): st.markdown(user_input)
    with st.chat_message("assistant"):
        response = chain.invoke(user_input)
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
