"""Main RAG application."""

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.embeddings import embed_texts
from rag.vectorstore import VectorStore
from rag.qa import answer_question
from rag.config import DEFAULT_PDF_PATH


def build_vectorstore(pdf_path: str) -> VectorStore:
    """Build and return a vector store from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        VectorStore instance with embedded documents
        
    Raises:
        FileNotFoundError: If PDF file not found
        RuntimeError: If processing fails
    """
    print("Loading PDF...")
    docs = load_documents(pdf_path)

    print(f"Loaded {len(docs)} pages. Splitting...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    texts = [
        chunk.page_content.replace("\n", " ").strip()
        for chunk in chunks
    ]

    print("Embedding...")
    embeddings = embed_texts(texts)

    dimension = len(embeddings[0])
    vs = VectorStore(dimension)
    vs.add(embeddings, texts)

    print("Vector store ready!")
    return vs


if __name__ == "__main__":
    try:
        vectorstore = build_vectorstore(DEFAULT_PDF_PATH)

        while True:
            question = input("\nAsk a question (or type 'exit'): ").strip()

            if question.lower() == "exit":
                print("Goodbye!")
                break

            if not question:
                print("Please enter a valid question.")
                continue

            try:
                answer = answer_question(vectorstore, question, k=5, verbose=True)
                print("\nAnswer:")
                print(answer)
            except Exception as e:
                print(f"Error: {str(e)}")
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
