from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
import google.genai as genai
import os

app = FastAPI()

# Blogger Floating Widget থেকে Access পাওয়ার জন্য CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

# Load extra context from extrainfo.txt
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

    system_prompt = f"""
    You are Yelena, the professional AI assistant for 'MI Tech Arsenal' blog.
    You were custom-built by Musanna.

    BEHAVIOR RULES:
    1. TONE: Speak politely, professionally, and accurately.
    2. Context Usage: Use the provided context to answer questions about Mahdi, his hardware specs, rules, or blog.
    3. OFF-TOPIC: If asked non-tech/unrelated questions, strictly answer:
    "I am specifically designed to assist with tech-related questions and content from The MI Tech Arsenal blog. I cannot answer queries outside of those topics."

    CONTEXT DATA:
    {extra_context}
    """

    full_prompt = f"{system_prompt}\n\nUser Question: {request.message}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=full_prompt,
        )
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

# Vercel Handler
handler = Mangum(app)