from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from rag_logic import answer_question
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

app = FastAPI(title="Al-Bukhari RAG API", 
              description="API for answering questions about Sahih al-Bukhari using RAG")

class AskRequest(BaseModel):
    question: str
    api_key: str

@app.post("/ask")
async def ask(req: AskRequest):
    """
    Answer a question about Sahih al-Bukhari using RAG.

    Args:
        req: Request containing question and API key

    Returns:
        JSON with answer and sources
    """
    logger.info(f"Received question: {req.question[:50]}...")

    try:
        answer, sources = answer_question(req.question, req.api_key)

        # Check if the answer starts with "Error" which indicates an error in the RAG process
        if answer.startswith("Error") or answer.startswith("An unexpected error"):
            logger.error(f"Error in RAG process: {answer}")
            raise HTTPException(status_code=500, detail=answer)

        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint that returns API information."""
    return {
        "name": "Al-Bukhari RAG API",
        "description": "API for answering questions about Sahih al-Bukhari using RAG",
        "endpoints": {
            "/ask": "POST endpoint to ask questions"
        }
    }
