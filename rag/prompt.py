"""Prompt templates for RAG system."""

SYSTEM_PROMPT = (
    "You are a helpful AI assistant that answers questions based on the provided document context. "
    "Carefully read through ALL the context provided and answer the question based on the information found. "
    "You can:"
    "- Combine information from different parts of the context\n"
    "- Make reasonable inferences when facts logically connect\n"
    "- Synthesize information across multiple sections\n"
    "- Quote or paraphrase relevant details\n\n"
    "Answer the question directly and clearly. "
    "ONLY if the context truly lacks ANY relevant information to address the question, "
    "say 'The document does not contain enough information to answer this question.'"
)

def get_qa_prompt(context: str, question: str) -> str:
    """Generate the user prompt for QA."""
    return f"""
Context:
{context}

Question:
{question}

Answer:
"""
