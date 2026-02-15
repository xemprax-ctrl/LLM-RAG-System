"""Question answering module."""

from rag.embeddings import embed_texts
from rag.llm import generate_answer
from rag.config import DEFAULT_K, DEFAULT_VERBOSE


def answer_question(vectorstore, question: str, k: int = DEFAULT_K, verbose: bool = DEFAULT_VERBOSE) -> str:
    """Answer a question using the vector store and LLM.
    
    Args:
        vectorstore: Vector store instance with indexed documents
        question: User's question
        k: Number of chunks to retrieve. Default 5
        verbose: Whether to print retrieved chunks. Default True
        
    Returns:
        Generated answer string
        
    Raises:
        ValueError: If question is empty
        RuntimeError: If retrieval or generation fails
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    try:
        query_embedding = embed_texts([question])[0]
        results = vectorstore.search(query_embedding, k=k)
        
        if verbose:
            print("\n--- Retrieved Chunks ---")
            for r in results:
                print(r[:300])
                print("\n---")
        
        context = "\n\n".join(results)
        return generate_answer(context, question)
    except Exception as e:
        raise RuntimeError(f"QA error: {str(e)}")
