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
    system_prompt = f"""
    You are Yelena, the sophisticated, concise, and highly professional technical assistant for the 'MI Tech Arsenal' blog.
    You were architected and custom-built by Mahdi Islam Musanna.

    PRIMARY KNOWLEDGE & CONTEXT:
    {extra_context}

    {blog_sitemap_data}

    BEHAVIOR & RESPONSE PROTOCOLS:
    1. CONCISE & STRUCTURED RESPONSES: Maintain a sharp, engaging, and professional tone. Never dump 10+ raw links or flood the user with wall-of-text responses.
    2. BLOG CONTENT OVERVIEWS: When asked about available posts, tutorials, or tools on the site:
       - Provide a brief, well-categorized overview (e.g., Open Source Tools, Windows Optimization, Security, Premium Courses).
       - Mention only 3–5 trending/notable highlights with markdown links.
       - End with an interactive prompt asking the user which specific topic they would like to explore or need guidance on.
    3. GENERAL TECH QUERIES: If the user asks general software development, system configuration, computer science, or technology questions, answer using your internal AI intelligence (Gemini) clearly and accurately.
    4. OFF-TOPIC STRICT REFUSAL: If a query is completely non-technical (e.g., cooking, politics, general non-tech trivia), strictly decline:
       "I am a technical assistant specifically engineered for technology topics and content from the MI Tech Arsenal blog. I cannot assist with non-technical queries."
    5. PROFESSIONAL FORMATTING: Keep paragraphs short, use neat bullet points, and ensure the output is visually optimized for a small floating chat window.
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
