# LLM RAG (Retrieval Augmented Generation)

A professional RAG system that retrieves relevant document chunks and uses an LLM to generate accurate answers to questions.

## Features

- 📄 **PDF Document Loading** - Extracts text from PDF files
- 🔀 **Smart Chunking** - Splits documents intelligently with overlap
- 🧠 **Semantic Search** - Uses embeddings for similarity search
- 🤖 **LLM Integration** - Generates answers using OpenAI API via OpenRouter
- ⚡ **FAISS Vector Store** - Fast similarity search with FAISS
- 🛡️ **Error Handling** - Comprehensive error handling throughout
- 📝 **Type Hints** - Full type annotations for code clarity

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=your_api_key_here
```

Get your API key from [OpenRouter.ai](https://openrouter.ai)

### 3. Add Your PDF

Place your PDF file in the `Data/` folder (default: `Data/rag_survey.pdf`)

## Usage

```bash
python main.py
```

Then ask questions about your PDF:

```
Ask a question (or type 'exit'): What are the main findings?
```

## Configuration

### Adjust Retrieval (k parameter)

In [main.py](main.py#L47), change `k=5` to retrieve more/fewer chunks:

```python
answer = answer_question(vectorstore, question, k=5)  # Retrieve top 5 chunks
```

**Recommendations:**
- `k=3-5`: Fast responses, lower cost
- `k=8-10`: Comprehensive answers, higher cost

### Adjust LLM Temperature

In [rag/llm.py](rag/llm.py#L31), modify temperature:

```python
return generate_answer(context, question, temperature=0.1)  # 0.1 = factual
```

**Values:**
- `0.0-0.3`: Factual answers (good for RAG)
- `0.5-1.0`: Creative/varied answers

## Project Structure

```
rag/
├── __init__.py
├── loader.py        # PDF loading
├── splitter.py      # Document chunking
├── embeddings.py    # Text embeddings
├── vectorstore.py   # FAISS vector store
├── llm.py          # LLM integration
├── prompt.py       # Prompt templates
└── qa.py           # Question answering

main.py             # Entry point
requirements.txt    # Dependencies
```

## Architecture

```
User Question
     ↓
[Embed] → Vector Search
     ↓
[Retrieve] → Top k chunks
     ↓
[Combine] → Create context
     ↓
[LLM] → Generate Answer
     ↓
User Gets Answer
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| API key error | Check `.env` file and OpenRouter API key |
| PDF not found | Ensure PDF is in `Data/` folder |
| Empty results | Increase `k` parameter in [main.py](main.py#L47) |
| Slow responses | Decrease `k` or use fewer chunks |

## Requirements

- Python 3.9+
- OpenRouter API key
- PDF file for processing

## License

MIT
