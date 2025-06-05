import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from typing import Optional, List

# Load environment variables
load_dotenv()
assert "OPENAI_API_KEY" in os.environ, "❌ OPENAI_API_KEY is missing. Required for loading FAISS embeddings."
assert "OPENROUTER_API_KEY" in os.environ, "❌ OPENROUTER_API_KEY is missing. Set it as an environment variable."

# Path to the FAISS index (relative for Render)
DB_PATH = os.getenv("FAISS_PATH", "./faiss_index")

# Load FAISS vectorstore
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local(
    folder_path=DB_PATH,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever()

# Custom wrapper for DeepSeek via OpenRouter
class OpenRouterLLM(LLM):
    model: str = "deepseek/deepseek-chat-v3-0324:free"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = os.getenv("OPENROUTER_API_KEY")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            extra_headers={
                "X-Title": "Ask al-Bukhari",
            }
        )
        return response.choices[0].message.content

    @property
    def _llm_type(self) -> str:
        return "openrouter"

llm = OpenRouterLLM()

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Streamlit frontend
st.set_page_config(page_title="Ask al-Bukhari", layout="wide")
st.title("📖 Ask al-Bukhari — RAG Question Answering")

query = st.text_input("Your query:")

if query:
    with st.spinner("Searching for answer..."):
        try:
            result = qa({"query": query})
            st.markdown("### Answer:")
            st.write(result["result"])
            st.markdown("### Source Hadith(s):")
            for i, doc in enumerate(result["source_documents"], 1):
                st.markdown(f"**{i}.** {doc.page_content}")
                meta = doc.metadata
                st.markdown(f"*Volume:* `{meta.get('volume', '')}` | *Book:* `{meta.get('book', '')}` | *Number:* `{meta.get('number', '')}` | *Narrator:* `{meta.get('narrator', '')}`")
                st.markdown("---")
        except Exception as e:
            st.error(f"Error processing the query: {e}")