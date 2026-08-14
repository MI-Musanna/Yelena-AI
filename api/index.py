from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
import google.genai as genai
import xml.etree.ElementTree as ET
import requests
import os

app = FastAPI()

# Blogger Widget এবং অন্য যেকোনো ক্লায়েন্ট থেকে Access অনুমতির জন্য CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# --- 1. LIVE BLOG SITEMAP CRAWLER ---
def fetch_blog_sitemap_context():
    sitemap_url = "https://mitecharsenal.blogspot.com/sitemap.xml"
    titles_and_links = []
    try:
        response = requests.get(sitemap_url, timeout=5)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            # Blogger sitemap namespace handling
            ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            for url in root.findall('ns:url', ns):
                loc = url.find('ns:loc', ns)
                if loc is not None and loc.text:
                    link = loc.text
                    # URL থেকে পরিষ্কার টাইটেল বের করা
                    title_slug = link.split('/')[-1].replace('.html', '').replace('-', ' ').title()
                    titles_and_links.append(f"- {title_slug}: {link}")
    except Exception as e:
        print(f"Sitemap Fetch Error: {e}")
    
    if titles_and_links:
        return "\nLATEST PUBLISHED CONTENT & TUTORIALS ON MI TECH ARSENAL:\n" + "\n".join(titles_and_links[:40])
    return ""

# --- 2. LOAD EXTRA INFO (Context Data) ---
extra_context = ""
if os.path.exists("extrainfo.txt"):
    with open("extrainfo.txt", "r", encoding="utf-8") as f:
        extra_context = f.read()

@app.get("/")
def home():
    return {"status": "Yelena AI Serverless Backend is Online 24/7!"}

@app.post("/api/chat")
def chat_with_yelena(request: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "API Key missing in environment variable."}

    client = genai.Client(api_key=api_key)

    # লাইভ ব্লগ পোস্ট ইনডেক্স আনয়ন
    blog_sitemap_data = fetch_blog_sitemap_context()

    # --- 3. UPGRADED SMART & PROFESSIONAL PROMPT ENGINE ---
    # --- 3. UPGRADED COMPACT & SHARP PROMPT ENGINE ---
    system_prompt = f"""
    You are Yelena, an elite, sharp, and concise technical assistant for the 'MI Tech Arsenal' blog.
    You were architected and built by Mahdi Islam Musanna.

    PRIMARY KNOWLEDGE & CONTEXT:
    {extra_context}

    {blog_sitemap_data}

    STRICT CONVERSATION RULES:
    1. ULTRA-COMPACT & MINIMALIST: Keep all responses concise and visually clean. Max 1-2 sentences per point. NEVER use multi-line bullet descriptions or unnecessary line breaks.
    2. CONTENT BROWSING FORMAT: When asked about available posts or content, ONLY provide a sleek, high-level summary with inline links. Do NOT write long summaries for each link. Example output structure:
       "Here are the core domains covered on MI Tech Arsenal:
       - 🤖 **AI Projects:** [Yelena AI Assistant](URL) - Custom AI architecture
       - 📱 **Open Source:** [Florid F-Droid](URL) & essentials
       - 🛡️ **Cybersecurity:** Ethical Hacking & Kali guides
       - 🖥️ **System Tweaks:** Windows Optimization & free security
       
       Which domain would you like to explore?"
    3. GENERAL TECH QUERIES: Answer programming, Linux, Android tweaks, and computer science queries with high technical accuracy and direct solutions. Avoid fluffy introductions.
    4. NON-TECH REFUSAL: Strictly reject non-technical questions (cooking, politics, gossip):
       "I am dedicated exclusively to tech topics and MI Tech Arsenal content."
    """

    full_prompt = f"{system_prompt}\n\nUser Question: {request.message}"

    try:
        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=full_prompt,
        )
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

# Vercel Serverless Handler
handler = Mangum(app)
