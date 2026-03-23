# ⚡ Yelena AI Assistant
**Automated RAG-Based Knowledge Engine for [MI Tech Arsenal](https://mitecharsenal.blogspot.com/)**

Yelena is a custom-built AI agent designed to act as a 24/7 technical assistant. Using Retrieval-Augmented Generation (RAG), she actively crawls 90+ blog posts, processes complex technical queries, and delivers precise answers based on published expertise.

---

## 🚀 Core Features
* **Live Sitemap Sync:** Automatically discovers and indexes new blog posts via `sitemap.xml`.
* **Neural Search:** Leverages Sentence Transformers to understand the *meaning* behind questions.
* **Floating Web Integration:** Deployed as a persistent, mobile-responsive widget directly into Blogger.
* **Context-Aware:** Fully briefed on admin rules, hardware specs, and software workflows.

---

## 🛠️ Tech Stack
* **Language:** Python 3.14
* **LLM:** Google Gemini 2.5 Flash
* **Vector Database:** ChromaDB (In-Memory)
* **Embeddings:** all-MiniLM-L6-v2
* **Framework:** Streamlit

---

## 💻 Quick Install

To run this RAG pipeline on your local machine, follow these steps:

**1. Clone the repository**
```bash
git clone [https://github.com/LittleEagle2007/Yelena-AI.git](https://github.com/LittleEagle2007/Yelena-AI.git)
cd Yelena-AI
```

**2. Activate Virtual Environment**
```bash
python -m venv ai_env
ai_env\Scripts\activate
```

**3. Install Requirements**
```bash
pip install -r requirements.txt
```

**4. Launch Yelena**
```bash
streamlit run app.py
```

---

## 👨‍💻 System Architect

**Mahdi Islam (Musanna)**
* 🎓 CST Student @ Daffodil Polytechnic Institute 
* 💻 Hardware: Intel i5 12400F, RX6600, 16GB RAM
* 🔗 [Visit MI Tech Arsenal](https://mitecharsenal.blogspot.com/) | [GitHub Portfolio](https://github.com/LittleEagle2007)

> Built with precision to bridge the gap between technical content and user accessibility.
