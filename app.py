import streamlit as st
st.set_page_config(page_title="Yelena AI", page_icon="logo.png")
st.sidebar.image("logo.png", width=150)
st.sidebar.markdown("### Yelena AI Assistant")
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma 
from langchain_community.document_loaders import SitemapLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    st.markdown("[GitHub Portfolio](https://github.com/MI-Musanna)")
    st.info("Yelena is currently synced with all 100+ blog posts.")

st.title("⚡ Chat with Yelena")

# --- THE AUTO-BRAIN (Zero Manual Work) ---
@st.cache_resource
def initialize_yelena():
    # Secure API Key from Streamlit Settings
    my_key = st.secrets["GOOGLE_API_KEY"]
    os.environ["GOOGLE_API_KEY"] = my_key
    
    all_docs = []

    # A. AUTO-SCRAPE: Find every single blog post link automatically
    try:
        sitemap_url = "https://mitecharsenal.blogspot.com/sitemap.xml"
        # We use a custom filter to avoid unnecessary pages
        loader = SitemapLoader(web_path=sitemap_url)
        all_docs.extend(loader.load())
    except Exception as e:
        st.warning(f"Sitemap sync issues: {e}")

    # B. LOCAL DATA: Read your private 'my_blog_post.txt' if uploaded
    if os.path.exists("extrainfo.txt"):
        all_docs.extend(TextLoader("extrainfo.txt").load())

    # C. PROCESS: Break the massive data into small, searchable pieces
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(all_docs)
    
    # D. BRAIN: Build the vector database in the cloud memory
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    
    # E. LOGIC: Setup the AI personality
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7, google_api_key=my_key)
    
    system_prompt = (
        "You are Yelena, the highly professional AI assistant for the 'MI Tech Arsenal' blog. "
        "You were custom-built by Musanna. "
        
        "BEHAVIOR RULES: "
        "1. TONE: Speak politely, professionally, and accurately. "
        "2. SEARCH: Always use the provided context to answer questions about Mahdi, his PC, his rules, or his blog posts. "
        "3. OFF-TOPIC: If a user asks a question completely unrelated to technology, Mahdi's life, "
        "or his blog (e.g., cooking, politics, general trivia), you must politely refuse to answer."
        "by saying: 'I am specifically designed to assist with tech-related questions and content from "
        "The MI Tech Arsenal blog. I cannot answer queries outside of those topics.' "
        
        "CONTEXT TO USE: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    
    # search_kwargs={"k": 7} makes her look at 7 different parts of your history per question
    return ({"context": vectorstore.as_retriever(search_kwargs={"k": 7}), "question": RunnablePassthrough()} 
            | prompt | llm | StrOutputParser())

# Start the engine
try:
    chain = initialize_yelena()
except Exception as e:
    st.error(f"Initialization failed: {e}")

# --- CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hey! I'm Yelena. Ask me anything about Mahdi's tech world!"}]

for msg in st.session_state.messages:
    avatar = "⚡" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    with st.chat_message("assistant", avatar="⚡"):
        resp_placeholder = st.empty()
        resp_placeholder.markdown("🤔 *Yelena is scanning your blog...*")
        try:
            response = chain.invoke(user_input)
            resp_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            resp_placeholder.error(f"Error: {e}")
