# 📖 Ask al-Bukhari — RAG Question Answering App

This is a simple Retrieval-Augmented Generation (RAG) web app built with [Streamlit](https://streamlit.io/) and [LangChain](https://www.langchain.com/), using a FAISS vector index and the free `deepseek-chat-v3-0324` model via [OpenRouter](https://openrouter.ai/).

---

## 🧠 Features

- Semantic question answering over a curated hadith corpus (Al-Bukhari)
- FAISS-based vector retrieval
- OpenAI embeddings (locally stored, no runtime cost)
- DeepSeek LLM via OpenRouter (completely free)
- Streamlit-based interface

---

## 🚀 Live Deployment (Render)

To deploy on [Render](https://render.com/):

1. Push this project to a GitHub repository
2. Create a new **Web Service** on Render
3. Set the root directory to the project folder
4. Add the following environment variables manually:

OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key

5. Deploy using the included `render.yaml`.

---

## 💻 Local Setup

### Prerequisites

- Python 3.9+
- `pip`

### Installation

```bash
git clone https://github.com/yourusername/al-bukhari-rag.git
cd al-bukhari-rag
pip install -r requirements.txt

Configure your environment

Create a .env file and add your API keys:

OPENAI_API_KEY=your_openai_key
OPENROUTER_API_KEY=your_openrouter_key

Run the app

streamlit run app.py

Then open http://localhost:8501 in your browser.

⸻

📂 Project Structure

.
├── app.py               # Main Streamlit app
├── faiss_index/         # FAISS index with OpenAI embeddings
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
├── render.yaml          # Render deployment config
├── README.md
└── .gitignore


⸻

🛡️ Privacy & Cost Notes
	•	FAISS index is loaded locally. No OpenAI requests are made during query time.
	•	DeepSeek is called via OpenRouter and is entirely free at the time of writing.
	•	You only need OPENAI_API_KEY for loading the index, not for runtime inference.

⸻

📬 License

MIT — Free to use and modify.

---