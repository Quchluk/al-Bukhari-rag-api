import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# Загрузи FAISS индекс
def load_vectorstore():
    with open("hadith_index/index.pkl", "rb") as f:
        store = pickle.load(f)
    return store


# Главная функция ответа
def answer_question(question: str, api_key: str):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key, base_url="https://openrouter.ai/api/v1")
    db = FAISS.load_local("hadith_index", embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(
    openai_api_key=api_key,
    model_name="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1"
)

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = chain({"query": question})

    docs = retriever.get_relevant_documents(question)
    sources = [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]

    return result["result"], sources
