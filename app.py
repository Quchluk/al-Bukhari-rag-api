import os
from pathlib import Path
from typing import Optional, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM

# ─────────────────────────────────────────────────────────────────────
# Streamlit page setup  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Ask al-Bukhari", page_icon="📖", layout="wide")

# Minimalist CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@300;600&display=swap');
    html, body, div, input, textarea, button, label {
        font-family: 'Libre Franklin', sans-serif;
    }
    .stButton>button {
        border: 2px solid #000 !important;
        background: transparent !important;
        color: #000 !important;
        font-weight: 600 !important;
        padding: 0.4rem 1.2rem !important;
        transition: background .2s, color .2s;
    }
    .stButton>button:hover {
        background: #000 !important;
        color: #fff !important;
    }
    #MainMenu, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────
# Environment & vector-store
# ─────────────────────────────────────────────────────────────────────
load_dotenv()
assert "OPENAI_API_KEY" in os.environ, "❌ OPENAI_API_KEY is missing."
assert "OPENROUTER_API_KEY" in os.environ, "❌ OPENROUTER_API_KEY is missing."

# Path to FAISS index
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("FAISS_PATH", BASE_DIR / "faiss_index")).expanduser().resolve()

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    folder_path=str(DB_PATH),
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)
retriever = vectorstore.as_retriever()

# ─────────────────────────────────────────────────────────────────────
# Custom LLM wrapper (DeepSeek via OpenRouter)
# ─────────────────────────────────────────────────────────────────────
class OpenRouterLLM(LLM):
    model: str = "deepseek/deepseek-chat-v3-0324:free"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = os.getenv("OPENROUTER_API_KEY")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={"X-Title": "Ask al-Bukhari"},
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "openrouter"

llm = OpenRouterLLM()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# ─────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────
st.title("Ask al-Bukhari")

st.caption(
    """
    **What is this?** A semantic Q&A tool over *Sahih al-Bukhari* that returns
    an answer plus exact hadith sources.

    **How to use:**  
    1. Type your question.  
    2. Press **Enter** – you'll receive an answer and cited passages.
    """
)

query = st.text_input("Your query:")

if query:
    with st.spinner("Searching for answer…"):
        try:
            result = qa({"query": query})
            st.markdown("### Answer:")
            st.write(result["result"])

            st.markdown("### Source Hadith(s):")
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"**{i}.** {doc.page_content}")
                m = doc.metadata
                st.markdown(
                    f"*Volume:* `{m.get('volume','')}` | "
                    f"*Book:* `{m.get('book','')}` | "
                    f"*Number:* `{m.get('number','')}` | "
                    f"*Narrator:* `{m.get('narrator','')}`"
                )
                st.markdown("---")
        except Exception as e:
            st.error(f"Error processing the query: {e}")