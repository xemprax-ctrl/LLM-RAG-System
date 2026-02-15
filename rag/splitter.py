"""Document splitting module."""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.config import CHUNK_SIZE, CHUNK_OVERLAP, SEPARATORS


def split_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """Split documents into chunks.
    
    Args:
        documents: List of LangChain Document objects
        chunk_size: Size of each chunk in characters. Default 1000
        chunk_overlap: Overlap between chunks. Default 200
        
    Returns:
        List of document chunks
        
    Raises:
        ValueError: If documents list is empty or parameters invalid
    """
    if not documents:
        raise ValueError("Documents list cannot be empty")
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be between 0 and chunk_size")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS
    )

    chunks = splitter.split_documents(documents)
    return chunks
