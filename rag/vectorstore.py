"""Vector store module using FAISS."""

import faiss
import numpy as np


class VectorStore:
    """Vector store for semantic search using FAISS."""
    
    def __init__(self, dimension: int):
        """Initialize vector store.
        
        Args:
            dimension: Embedding dimension (e.g., 1536 for text-embedding-3-small)
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []

    def add(self, embeddings: list[list[float]], texts: list[str]) -> None:
        """Add embeddings and texts to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            texts: List of corresponding text chunks
            
        Raises:
            ValueError: If embeddings and texts length mismatch
        """
        if len(embeddings) != len(texts):
            raise ValueError("Embeddings and texts must have same length")
        
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.texts.extend(texts)

    def search(self, query_embedding: list[float], k: int = 3) -> list[str]:
        """Search for similar texts.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of k most similar text chunks
            
        Raises:
            ValueError: If k is invalid
        """
        if k <= 0 or k > len(self.texts):
            raise ValueError(f"k must be between 1 and {len(self.texts)}")
        
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        return [self.texts[i] for i in indices[0]]
