import pickle
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# Загрузи FAISS индекс
def load_vectorstore():
    with open("hadith_index/index.pkl", "rb") as f:
        store = pickle.load(f)
    return store


# Главная функция ответа
def answer_question(question: str, api_key: str):
    embeddings = OpenAIEmbeddings(
<<<<<<< HEAD
        openai_api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
    
=======
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

>>>>>>> 8bff9957 (fix: switch to langchain-openai with gpt-4o-mini)
    db = FAISS.load_local("hadith_index", embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
<<<<<<< HEAD
        openai_api_key=api_key,
        model_name="gpt-4o-mini", 
=======
        api_key=api_key,
        model="gpt-4o-mini",
>>>>>>> 8bff9957 (fix: switch to langchain-openai with gpt-4o-mini)
        base_url="https://openrouter.ai/api/v1"
    )

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    result = chain({"query": question})

    docs = retriever.get_relevant_documents(question)
    sources = [{"text": doc.page_content, "metadata": doc.metadata} for doc in docs]

    return result["result"], sources