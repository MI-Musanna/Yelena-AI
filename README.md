# ⚡ Yelena AI Assistant
**Automated Knowledge Engine for [MI Tech Arsenal](https://mitecharsenal.blogspot.com/)**

Yelena is a custom-built AI agent designed to act as a 24/7 technical assistant. Deployed as a serverless API, she processes technical queries and delivers precise answers based on published expertise and custom knowledge bases.

---

## 🚀 Core Features
* **24/7 Serverless Uptime:** Hosted on Vercel with zero app hibernation or sleep delay.
* **FastAPI Backend:** Lightweight and high-performance API endpoint for instant responses.
* **Floating Web Integration:** Deployed as a persistent, mobile-responsive widget directly into Blogger.
* **Context-Aware:** Fully briefed on admin rules, hardware specs, and software workflows.

---

## 🛠️ Tech Stack
* **Language:** Python
* **LLM:** Google Gemini 2.5 Flash
* **Framework:** FastAPI
* **Handler:** Mangum (ASGI Adapter)
* **Deployment Platform:** Vercel (Serverless Functions)

---

## 💻 API Usage & Local Setup

To run this RAG pipeline on your local machine, follow these steps:

### 1. Run Locally

**Clone the repository:**
```bash
git clone [https://github.com/MI-Musanna/Yelena-AI.git](https://github.com/MI-Musanna/Yelena-AI.git)
cd Yelena-AI
'''

```bash
uvicorn api.index:app --reload
