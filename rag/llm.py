"""LLM integration module using OpenRouter."""

import os
from openai import OpenAI
from dotenv import load_dotenv
from rag.prompt import SYSTEM_PROMPT, get_qa_prompt
from rag.config import LLM_MODEL, TEMPERATURE

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


def generate_answer(context: str, question: str, temperature: float = TEMPERATURE) -> str:
    """Generate an answer to a question based on provided context.
    
    Args:
        context: Document context retrieved from vector store
        question: User's question
        temperature: Sampling temperature (0.0-1.0). Default 0.1 for factual answers
        
    Returns:
        Generated answer string
        
    Raises:
        ValueError: If context or question is empty
        RuntimeError: If API call fails
    """
    if not context or not question:
        raise ValueError("Context and question cannot be empty")
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": get_qa_prompt(context, question)}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"LLM API error: {str(e)}")
