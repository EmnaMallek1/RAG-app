# AI Research Paper Assistant

A Retrieval-Augmented Generation (RAG) application for querying Artificial Intelligence and Machine Learning research papers and generating answers grounded in their content.

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-green)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Database-purple)](https://www.trychroma.com/)

## Live Demo

[Open the application](https://rag-app-phdg5jgnpytrykfcai3ge6.streamlit.app/)

## About

AI Research Paper Assistant allows users to ask questions about a collection of 26 AI and Machine Learning research papers.

The application uses a RAG pipeline to retrieve relevant information from the papers and generate answers using an LLM, while providing the corresponding sources.

## Features

- Semantic search across 26 AI/ML research papers
- Vector search with ChromaDB
- Cross-Encoder reranking
- LLM-based answer generation
- Source citations with paper names and pages
- Optional answer verification to reduce hallucinations
- Conversation history
- User authentication
- Dark and light themes

## Architecture

```text
Research Papers
       ↓
Text Chunking
       ↓
Embeddings
       ↓
ChromaDB
       ↓
Semantic Retrieval
       ↓
Cross-Encoder Reranking
       ↓
Llama 3.3 70B
       ↓
Answer + Sources
```
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
