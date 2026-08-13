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
* **LLM:** Google Gemini 3.5 Flash
* **Framework:** FastAPI
* **Handler:** Mangum (ASGI Adapter)
* **Deployment Platform:** Vercel (Serverless Functions)

---

## 💻 API Usage & Local Setup

### 1. Run Locally

**Step 1: Clone the repository**
```bash
git clone https://github.com/MI-Musanna/Yelena-AI.git
cd Yelena-AI
```

**Step 2: Install Requirements**
```bash
pip install -r requirements.txt
```

**Step 3: Launch Local Server**
```bash
uvicorn api.index:app --reload
```

---

### 2. API Endpoint

Once deployed to Vercel, send a `POST` request to `/api/chat`:

```json
{
  "message": "What are Mahdi's PC specifications?"
}
```

---

## 👨‍💻 System Architect

**Mahdi Islam (Musanna)**
* 🎓 CST Student @ Daffodil Polytechnic Institute 
* 💻 Hardware: Intel i5 12400F, RX6600, 16GB RAM
* 🔗 [Visit MI Tech Arsenal](https://mitecharsenal.blogspot.com/) | [GitHub Portfolio](https://mi-musanna.github.io)

> Built with precision to bridge the gap between technical content and user accessibility.
