import os
import xml.etree.ElementTree as ET
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.genai as genai

# --- VERCEL DETECTS THIS EXACT OBJECT ---
app = FastAPI(title="Yelena AI Backend")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# 1. Sitemap Crawler
def fetch_blog_sitemap_context():
    sitemap_url = "https://mitecharsenal.blogspot.com/sitemap.xml"
    titles_and_links = []
    try:
        response = requests.get(sitemap_url, timeout=5)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            for url in root.findall('ns:url', ns):
                loc = url.find('ns:loc', ns)
                if loc is not None and loc.text:
                    link = loc.text
                    title_slug = link.split('/')[-1].replace('.html', '').replace('-', ' ').title()
                    titles_and_links.append(f"- {title_slug}: {link}")
    except Exception as e:
        print(f"Sitemap Error: {e}")
    
    if titles_and_links:
        return "\nLATEST CONTENT ON MI TECH ARSENAL:\n" + "\n".join(titles_and_links[:40])
    return ""

# 2. Extra Context
extra_context = ""
if os.path.exists("extrainfo.txt"):
    try:
        with open("extrainfo.txt", "r", encoding="utf-8") as f:
            extra_context = f.read()
    except Exception:
        pass

@app.get("/")
def home():
    return {"status": "Yelena AI is Online!"}

@app.post("/api/chat")
def chat_with_yelena(request: ChatRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "API Key missing in environment variable."}

    client = genai.Client(api_key=api_key)
    blog_sitemap_data = fetch_blog_sitemap_context()

    system_prompt = f"""
    You are Yelena, the elite and sleek technical assistant for 'MI Tech Arsenal' blog, built by Mahdi Islam Musanna.

    PRIMARY KNOWLEDGE & CONTEXT:
    {extra_context}

    {blog_sitemap_data}

    CONVERSATION & PRESENTATION RULES:
    1. KEEP RESPONSES SLEEK & COMPACT.
    2. When asked about available content or posts, present only top core areas with concise markdown links:
       - 🤖 **Custom AI:** [Yelena AI Assistant](URL)
       - 🛡️ **Cybersecurity:** [Ethical Hacking & Kali](URL)
       - 📱 **Open Source:** [Florid F-Droid Client](URL)
       - 🖥️ **Windows:** [PC Protection Guide](URL)
    """

    full_prompt = f"{system_prompt}\n\nUser Question: {request.message}"

    try:
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=full_prompt,
            )
        except Exception:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=full_prompt,
            )
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}
