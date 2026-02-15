"""Text embedding module using OpenRouter."""

import os
from openai import OpenAI
from dotenv import load_dotenv
from rag.config import EMBEDDING_MODEL

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        List of embedding vectors
        
    Raises:
        ValueError: If texts list is empty
        RuntimeError: If API call fails
    """
    if not texts:
        raise ValueError("Texts list cannot be empty")
    
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        raise RuntimeError(f"Embedding API error: {str(e)}")
