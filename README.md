# Tutor_Bot---Retrieval-Augmented Generation (RAG)-Assistant
Tutor_Bot is a local, retrieval-augmented question-answer assistant designed for learning:
- Python fundamentals,
- data structures and algorithms in Python.

In this specific use case, instead of answering from the general internet or from the model's pretraining, the tutor only answers using a small curated set of textbooks that you provide (for example, open-licensed books like "A First Course on Data Structures in Python" by Donald R. Sheehy, MIT License © 2019 Don Sheehy).

If the answer is not present in the ingested sources/files, the tutor will explicitly mention:
"I do not have enough information in the provided documents."

This makes it useful for focused self-study,classroom settings, and portfolio demonstrations of Retrieval-Augmented Generation (RAG).

---

## ==============================  Features ==============================

- **Local / offline** inference using [Ollama](https://ollama.ai) and a lightweight model (Phi-3 3.8B).
- **Hugging Face Embeddings:** Uses 'BAAI/bge-small-en-v1.5' for sentence-level semantic similarity. This balances retrieval quality and speed for educational documents.
- **RAG pipeline**:
   user question --> retrieve top chunks from PDF (Portable Document Format) sources via FAISS ( Facebook AI Similarity Search) --> pass both question + context to the model.
- **Citations**: answers include '[book.pdf, pX]' so learners can verify claims.
- **Refrain policy**: the tutor refrains to answer ("I do not have enough information in the provided documents."):
  - if the question is off-topic 
  - or if the information is not found in the PDFs.
- **Deterministic** responses ('temperature=0.0') to reduce hallucinations.
- **Short/Long control**: you can ask "in 60 words", "detailed", "short", etc. Word budget is then enforced with soft trimming.

## ==============================  How It Works? ==============================

1. **Ingest Phase ('ingest.py')**
   - Reads all PDF files from './data/'.
   - Splits each PDF into overlapping text chunks (configurable size).
   - Stores each chunk with metadata (source filename, page number)
   - Embeds chunks using a sentence embedding model (default: `BAAI/bge-small-en-v1.5`).
       **Why this model?**
     - 'BAAI/bge-small-en-v1.5' is a compact, high-quality English text embedding model.  
     - It works offline and provides semantic similarity strong enough for retrieval tasks.  
     - 'normalize_embeddings=True' ensures all vectors are normalized, improving FAISS search accuracy.
     - Builds a FAISS vector index from these embeddings and saves it to './storage/faiss_index/'.
2. **Chat Phase ('cli_tutor.py')**
   - For input you type a question.
   - The app retrieves the most relevant chunks from FAISS.
   - The local LLM (running in Ollama) generates an answer using only the context provided by the user.
   - The app trims the answer to the requested word budget/limiting phrase and shows citations wherever possible.

-----------------------------------

## Directory Structure

python-tutor/
├─ README.md
├─ LICENSE
├─ .gitignore
├─ requirements.txt
├─ .env.example

├─ src/
│  ├─ ingest.py        # builds the FAISS index from PDFs
│  ├─ tutor_cli.py     # command line interactive tutor

├─ data/
│  ├─ README_DATA.md   # instructions for adding PDFs
│  └─ sample_docs/     # example pdf, open-licensed documents only

└─ storage/
   └─ faiss_index/     # generated vector store (NOT committed)





