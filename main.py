from fastapi import FastAPI, Request
from pydantic import BaseModel
from rag_logic import answer_question

app = FastAPI()

class AskRequest(BaseModel):
    question: str
    api_key: str

@app.post("/ask")
async def ask(req: AskRequest):
    answer, sources = answer_question(req.question, req.api_key)
    return {"answer": answer, "sources": sources}