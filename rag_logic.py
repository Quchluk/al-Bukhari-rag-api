from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


def answer_question(question: str, api_key: str):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
        base_url="https://openrouter.ai/v1",
    )

    db = FAISS.load_local(
        "hadith_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model_name="gpt-4o-mini",
        openai_api_key=api_key,
        base_url="https://openrouter.ai/v1",
        temperature=0,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = [
        {"text": doc.page_content, "metadata": doc.metadata}
        for doc in result["source_documents"]
    ]

    return answer, sources