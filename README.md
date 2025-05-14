# 📄 Document Assistant

A powerful document Q&A system built with **LangChain** and **Google's Gemini AI**. This application allows you to upload PDF documents and ask natural language questions about their content using **Retrieval-Augmented Generation (RAG)**.

---

## 🎥 Document Assistant Demo

*Coming soon*

---

## ✨ Features

- 📄 **PDF Document Processing**: Upload and process multiple PDF files.
- 🔍 **Smart Document Search**: Uses **FAISS vector database**.
- 💬 **Question Answering**: Get precise answers with source citations.
- 💾 **Vector Store Management**: Save and load processed documents for future use.
- 🖥️ **User-Friendly Interface**: Clean **Streamlit** web interface with chat-style interaction.

---

## 🛠️ Technical Architecture

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Document Processing**: PDFs are loaded and split into manageable chunks.
2. **Embedding Generation**: Document chunks are converted to vector embeddings using **Sentence Transformers**.
3. **Vector Storage**: Embeddings are stored in a **FAISS vector database** for efficient similarity search.
4. **Question Answering**: User queries retrieve relevant document chunks which are sent to the **LLM**.
5. **Response Generation**: **Google's Gemini AI** generates answers based on the retrieved context.

---

## 🚀 Getting Started

### 📋 Prerequisites

- Python **3.8+**
- Google AI API key for the Gemini model

### 📥 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/document-assistant.git
cd document-assistant

# Install dependencies
pip install -r requirements.txt
