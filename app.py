import streamlit as st
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma 
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- UI SETUP ---
st.set_page_config(page_title="Yelena AI | Mahdi Islam", page_icon="⚡")

with st.sidebar:
    st.header("About the Dev")
    st.write("Built by Mahdi Islam")
    st.markdown("[Read my Blog](https://mitecharsenal.blogspot.com)")
    st.markdown("[GitHub Portfolio](https://github.com/LittleEagle2007)")

st.title("⚡ Chat with Yelena")
st.caption("The official AI assistant for Mahdi Islam's Blog.")

# --- BACKEND LOGIC ---
@st.cache_resource
def load_ai():
    # SECURE: Pulls the key from Streamlit's hidden vault
    my_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = my_key
    
    # Must match the model used in build_brain.py
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./blog_database", embedding_function=embeddings)
    
    system_prompt = (
        "You are Yelena, the sharp, tech-savvy human assistant for Mahdi Islam's blog. "
        "Rules: Speak like a human peer, stay brief, and use emojis. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8, google_api_key=my_key)
    
    # k=5 means she looks at more info, making her "smarter"
    chain = (
        {"context": db.as_retriever(search_kwargs={"k": 5}), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Initialize the chain
try:
    chain = load_ai()
except Exception as e:
    st.error(f"Setup Error: {e}")

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey! I'm Yelena. Ask me anything!"}]

for msg in st.session_state.messages:
    avatar = "⚡" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
        
    with st.chat_message("assistant", avatar="⚡"):
        resp_container = st.empty()
        resp_container.markdown("🤔 *Yelena is thinking...*")
        try:
            response = chain.invoke(user_input)
            resp_container.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            resp_container.error(f"Error: {e}")