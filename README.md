
# 📊 Natural Language Data Visualizer

**Natural Language Data Visualizer** is a powerful Retrieval-Augmented Generation (RAG) system that allows users to upload CSV datasets, ask natural language questions, and get insights through text answers and data visualizations. It combines a **Streamlit frontend**, **FastAPI backend**, and advanced language model pipelines using **LangChain**, **FAISS**, and **LLaMA2** to offer an interactive, intelligent interface for data exploration.

---

## 🚀 Project Overview

The application enables:
- CSV file upload and parsing
- Querying data using natural language
- Automatic metadata analysis and aggregation
- Context-aware visualizations (bar, line, scatter plots)
- Semantic search using FAISS and embeddings
- Answer generation using LLaMA2 via LangChain

---

## 🗂️ Project Structure

| File | Description |
|------|-------------|
| `streamlit_app.py` | Manages the user interface using Streamlit. Handles file uploads, natural language queries, and displays results and charts. |
| `app.py` | FastAPI backend providing endpoints for file upload, querying, and plot generation. Also manages CORS and API logic. |
| `main_agent.py` | Core logic for RAG pipeline: classifies queries, performs context retrieval, invokes LLM, and validates responses. |
| `data_processor.py` | Loads, stores, and manages CSV files using Pandas. Handles dataset listings, metadata, and parsing. |
| `visualizer.py` | Generates Matplotlib and Seaborn plots based on user input and column selection. |
| `conversation_memory.py` | Stores user query history and manages context-aware sessions (non-persistent). |

---

## 🔁 How the RAG System Works

1. **Query Understanding**: Queries are classified (e.g., metadata, aggregation, comparison).
2. **Context Retrieval**: Text chunks from datasets are vectorized using HuggingFace embeddings and searched with FAISS.
3. **Answer Generation**: LLaMA2 generates natural language answers using retrieved context.
4. **Response Validation**: Output is verified against actual dataset content for accuracy.

---

## 🧠 Key Technologies

- **LangChain**: Manages the entire RAG pipeline and flow between components.
- **LLaMA2**: The foundational large language model generating responses.
- **HuggingFace Embeddings**: Embeds text into vectors for semantic matching.
- **FAISS**: Enables fast similarity search and vector indexing.

---

## 📦 Chunking Strategy

- **Splitter**: `RecursiveCharacterTextSplitter`
- **Chunk Size**: 500 characters
- **Overlap**: 50 characters

This ensures contextual coherence while maintaining optimal chunk sizes for retrieval.

---

## 📥 CSV File Upload Process

1. File uploaded via Streamlit UI
2. Saved on backend under `/data/` directory
3. Parsed into Pandas DataFrame by `data_processor.py`
4. Embeddings generated using HuggingFace MiniLM
5. Stored in FAISS vector store for semantic search

---

## 🧭 Agent Responsibilities

- Classify user queries
- Retrieve relevant chunks using semantic similarity
- Use LLM to generate meaningful responses
- Validate and score responses
- Coordinate between retrieval, generation, and visualization

---

## 🔍 Retrieval Workflow

1. Embedding creation from dataset text
2. FAISS performs similarity search against query embedding
3. Top matches are returned as context for generation

---

## 📊 Visualizations

Visuals are generated on the backend using **Matplotlib** and **Seaborn**. The frontend allows selection of:
- Plot type (bar, line, scatter)
- X and Y axes (columns)

The backend renders the plot and streams it back to the Streamlit interface.

---

## 📈 Embedding Process

- **Model Used**: HuggingFace `all-MiniLM-L6-v2`
- Converts text to dense vector representations
- Stored in FAISS for fast, scalable semantic search

---

## 📋 Prompt Templates

Prompt templates guide the LLM using:
- Dataset metadata (e.g., column names, datatypes)
- Sample data from the uploaded CSV
- Contextual instructions based on query classification

---

## ❌ No SQL Conversion

CSV files are **not** converted into SQL. Instead, all operations are performed directly on Pandas DataFrames.

---

## 💬 Conversational Nature

While not stateful, the system supports conversational-style inputs and provides contextual, natural language answers.

---

## ✅ Query Relevance and Accuracy

Accuracy is ensured through:
- Smart chunking and retrieval
- Dataset-aware prompt construction
- Optional backend validation of answers

---

## ⚠️ Handling Wrong Answers

To improve incorrect outputs:
- Refine embedding and chunking
- Tune the prompt templates
- Add feedback mechanisms
- Add post-generation data validation

---

## ⚡ Performance Optimization Ideas

- Enable async endpoints and batch processing
- Optimize embedding/chunking operations
- Add Redis-based caching for frequent queries
- Replace LLaMA2 with faster models for prototyping

---

## 🌟 Future Enhancements

- Fully conversational memory (e.g., using Redis or ChromaDB)
- Multi-file or multi-sheet CSV support
- More plot types and advanced aggregations
- Admin interface for managing datasets
- Plug-and-play backend APIs for LLMs

---

## 📁 CSV Conversion Pipeline (Recap)

1. Upload → 2. Save → 3. Parse with Pandas  
4. Embed with HuggingFace → 5. Store in FAISS  
6. Retrieve based on query → 7. Generate response

---

## 🧠 Why this is one of the best apps for Dataset Querying

- Easy and intuitive interface
- LLM-powered querying
- Semantic understanding of raw data
- Flexible visualization support
- Scalable and customizable backend

---

## 🛠️ Requirements

- Python 3.9+
- FastAPI
- Streamlit
- Pandas
- FAISS
- LangChain
- Transformers (HuggingFace)
- Matplotlib, Seaborn

---

## 📌 Conclusion

The Natural Language Data Visualizer combines NLP, interactive UI, and ML pipelines to deliver a robust experience for dataset analysis. It bridges the gap between structured data and intuitive exploration using the power of large language models.

---
