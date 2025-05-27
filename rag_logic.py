from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

def answer_question(question: str, api_key: str):
    try:
        embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            base_url="https://openrouter.ai/v1"
        )

        db = FAISS.load_local("hadith_index", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 4})

        llm = ChatOpenAI(
            openai_api_key=api_key,
            model="openai/gpt-4o-mini",
            base_url="https://openrouter.ai/v1"
        )

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = chain({"query": question})
        sources = [
            {"text": doc.page_content, "metadata": doc.metadata}
            for doc in result.get("source_documents", [])
        ]

        return result["result"], sources

    except Exception as e:
        return f"Ошибка: {str(e)}", []