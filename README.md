# LLM-RAG-System

Retrieval-Augmented Generation (RAG) system for intelligent document processing and question answering using Large Language Models.

## Overview

This project implements a comprehensive Retrieval-Augmented Generation (RAG) system that combines the power of Large Language Models with document retrieval capabilities. It enables intelligent question-answering over custom documents by retrieving relevant context and generating accurate responses.

## Features

- **Document Processing**: Upload and process various document formats
- **Semantic Search**: Retrieve relevant documents based on semantic similarity
- **Question Answering**: Generate accurate answers based on retrieved context
- **LLM Integration**: Support for multiple LLM providers
- **Vector Storage**: Efficient vector database for document embeddings
- **API Support**: RESTful API endpoints for easy integration

## Project Structure

```
LLM-RAG/
├── rag/
│   ├── qa.py              # Question answering module
│   ├── retrieval.py       # Document retrieval module
│   ├── embeddings.py      # Embedding generation
│   └── utils.py           # Utility functions
├── main.py                # Main application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/xemprax-ctrl/LLM-RAG-System.git
cd LLM-RAG-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your settings:
```bash
# Edit config.py with your API keys and preferences
python config.py
```

## Usage

### Running the Application

```bash
python main.py
```

### Using as a Library

```python
from rag.qa import QuestionAnswerer

# Initialize the QA system
qa = QuestionAnswerer()

# Ask a question
answer = qa.answer("Your question here")
print(answer)
```

## API Endpoints

- `POST /api/upload` - Upload documents
- `POST /api/query` - Ask a question
- `GET /api/status` - Check system status

## Configuration

Edit `config.py` to configure:
- LLM provider (OpenAI, Hugging Face, etc.)
- Vector database settings
- Document processing options
- API credentials

## Dependencies

See `requirements.txt` for all dependencies. Main packages include:
- `langchain` - LLM framework
- `openai` - OpenAI API
- `faiss-cpu` - Vector database
- `sentence-transformers` - Embedding models
- `flask` - Web API framework

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

## Author

Created by Abdullah Alabdullah

---

**Last Updated**: February 15, 2026
