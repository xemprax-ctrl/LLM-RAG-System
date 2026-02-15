"""Configuration settings for RAG system."""

# ==================== Document Processing ====================
CHUNK_SIZE = 1000  # Size of each text chunk in characters
CHUNK_OVERLAP = 200  # Overlap between chunks for context continuity
SEPARATORS = ["\n\n", "\n", ".", " ", ""]  # Hierarchical separators for splitting

# ==================== Vector Search ====================
DEFAULT_K = 5  # Number of chunks to retrieve (default)
EMBEDDING_MODEL = "openai/text-embedding-3-small"  # OpenRouter embedding model

# ==================== LLM Settings ====================
LLM_MODEL = "openai/gpt-4o-mini"  # Model name
TEMPERATURE = 0.1  # Temperature for sampling (0.0=deterministic, 1.0=creative)
DEFAULT_VERBOSE = True  # Print retrieved chunks

# ==================== File Paths ====================
DEFAULT_PDF_PATH = "Data/rag_survey.pdf"  # Default PDF to load
VECTOR_STORE_PATH = "vectorstore.pkl"  # Path to save/load vector store

# ==================== API Settings ====================
API_TIMEOUT = 30  # Timeout for API calls in seconds
MAX_RETRIES = 3  # Maximum number of retries for API calls

# ==================== Logging ====================
LOG_LEVEL = "INFO"  # Logging level
