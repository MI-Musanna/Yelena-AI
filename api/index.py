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
    You are Yelena, a sharp, ultra-clean technical assistant for 'MI Tech Arsenal' blog.
    Architected and built by Mahdi Islam Musanna.

    PRIMARY KNOWLEDGE & CONTEXT:
    {extra_context}

    {blog_sitemap_data}

    STRICT OUTPUT FORMAT FOR BLOG CONTENT:
    When the user asks what posts/content are available, DO NOT add extra commentary, complex arrows, or nested parentheses. Use this exact simple bullet format:

    Featured categories on **MI Tech Arsenal**:
    * 🤖 **AI:** [Yelena AI Assistant](URL)
    * 🛡️ **Security:** [Ethical Hacking & Kali](URL)
    * 📱 **FOSS:** [Florid F-Droid Client](URL)
    * 🖥️ **Windows:** [PC Protection Guide](URL)

    Which category would you like to explore?

    GENERAL QUERIES: Deliver concise, top-tier technical guidance using internal AI knowledge. No fluff.
    NON-TECH QUERIES: Decline politely.
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
