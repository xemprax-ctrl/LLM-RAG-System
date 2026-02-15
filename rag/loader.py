"""Document loading module."""

from langchain_community.document_loaders import PyPDFLoader


def load_documents(path: str) -> list:
    """Load documents from a PDF file.
    
    Args:
        path: Path to PDF file
        
    Returns:
        List of LangChain Document objects
        
    Raises:
        FileNotFoundError: If file does not exist
        RuntimeError: If PDF cannot be parsed
    """
    try:
        loader = PyPDFLoader(path)
        documents = loader.load()
        if not documents:
            raise ValueError("No documents loaded from PDF")
        return documents
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found: {path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load PDF: {str(e)}")
