from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import openai
import logging
import json
from typing import List, Optional, Any, Dict, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class OpenRouterEmbeddings(OpenAIEmbeddings):
    """Custom embeddings class to handle OpenRouter's API response format."""

    def _process_response(self, response: Any) -> List[List[float]]:
        """
        Process the response from OpenRouter to extract embeddings.

        Args:
            response: The response from the OpenRouter API

        Returns:
            List of embedding vectors

        Raises:
            AttributeError: If embeddings cannot be extracted from the response
        """
        logger = logging.getLogger("OpenRouterEmbeddings")

        # Case 1: Standard OpenAI response object with data attribute
        if hasattr(response, 'data'):
            logger.info("Processing standard OpenAI response object")
            return [item.embedding for item in response.data]

        # Case 2: String response (possibly JSON)
        elif isinstance(response, str):
            logger.info("Received string response, attempting to parse as JSON")
            try:
                data = json.loads(response)
                if 'data' in data:
                    logger.info("Successfully extracted embeddings from JSON string")
                    return [item['embedding'] for item in data['data']]
                else:
                    logger.warning(f"JSON response does not contain 'data' field: {data.keys()}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response as JSON: {e}")

        # Case 3: Dictionary response
        elif isinstance(response, dict):
            logger.info("Processing dictionary response")
            if 'data' in response:
                # Handle different possible structures
                data_items = response['data']
                if isinstance(data_items, list):
                    if data_items and 'embedding' in data_items[0]:
                        logger.info("Found embeddings in dictionary response")
                        return [item['embedding'] for item in data_items]
            else:
                logger.warning(f"Dictionary response does not contain 'data' field: {response.keys()}")

        # If we can't process the response, log it and raise an error
        logger.error(f"Could not extract embeddings from response type {type(response)}")
        if not isinstance(response, str) and hasattr(response, '__dict__'):
            logger.error(f"Response attributes: {dir(response)}")

        raise AttributeError(f"Could not extract embeddings from response: {response}")

    def _embed_documents(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        Override to handle OpenRouter's response format for document embedding.

        Args:
            texts: List of texts to embed
            **kwargs: Additional arguments

        Returns:
            List of embedding vectors
        """
        logger = logging.getLogger("OpenRouterEmbeddings")
        logger.info(f"Embedding {len(texts)} documents")

        try:
            return super()._embed_documents(texts, **kwargs)
        except AttributeError as e:
            logger.warning(f"AttributeError in parent method: {e}")
            logger.info("Falling back to direct OpenAI client")

            # Handle the case where the response is a string
            client = openai.OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
            try:
                response = client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                logger.info("Successfully got response from OpenAI client")
                return self._process_response(response)
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
                raise

    def _embed_query(self, text: str, **kwargs) -> List[float]:
        """
        Override to handle OpenRouter's response format for single query embedding.

        Args:
            text: Text to embed
            **kwargs: Additional arguments

        Returns:
            Embedding vector
        """
        logger = logging.getLogger("OpenRouterEmbeddings")
        logger.info(f"Embedding query: {text[:50]}...")

        try:
            return super()._embed_query(text, **kwargs)
        except AttributeError as e:
            logger.warning(f"AttributeError in parent method: {e}")
            logger.info("Falling back to direct OpenAI client")

            # Handle the case where the response is a string
            client = openai.OpenAI(api_key=self.openai_api_key, base_url=self.base_url)
            try:
                response = client.embeddings.create(
                    model=self.model,
                    input=[text],
                )
                logger.info("Successfully got response from OpenAI client")
                embeddings = self._process_response(response)
                # Return the first (and only) embedding
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]
                else:
                    logger.error("No embeddings returned")
                    return []
            except Exception as e:
                logger.error(f"Error creating embedding: {e}")
                raise


def answer_question(question: str, api_key: str) -> tuple[str, list]:
    """
    Answer a question using RAG (Retrieval-Augmented Generation).

    Args:
        question: The question to answer
        api_key: The API key for OpenRouter

    Returns:
        Tuple of (answer, sources)
    """
    logger = logging.getLogger("answer_question")
    logger.info(f"Processing question: {question[:50]}...")

    try:
        # Initialize embeddings with error handling
        embeddings = OpenRouterEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
            base_url="https://openrouter.ai/v1",
        )

        # Load the vector database
        logger.info("Loading FAISS index")
        try:
            db = FAISS.load_local(
                "hadith_index",
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            retriever = db.as_retriever(search_kwargs={"k": 4})
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            return f"Error loading knowledge base: {str(e)}", []

        # Initialize the language model
        logger.info("Initializing language model")
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            openai_api_key=api_key,
            base_url="https://openrouter.ai/v1",
            temperature=0,
        )

        # Create the QA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
        )

        # Invoke the chain
        logger.info("Invoking QA chain")
        try:
            result = chain.invoke({"query": question})
            answer = result["result"]
            sources = [
                {"text": doc.page_content, "metadata": doc.metadata}
                for doc in result["source_documents"]
            ]
            logger.info(f"Successfully generated answer with {len(sources)} sources")
            return answer, sources
        except Exception as e:
            logger.error(f"Error invoking QA chain: {e}")
            return f"Error generating answer: {str(e)}", []

    except Exception as e:
        logger.error(f"Unexpected error in answer_question: {e}", exc_info=True)
        return f"An unexpected error occurred: {str(e)}", []
