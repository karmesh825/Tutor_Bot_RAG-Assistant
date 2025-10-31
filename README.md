# Tutor_Bot Retrieval-Augmented Generation (RAG)-Assistant
Tutor Bot is a local, retrieval-augmented question-answer assistant designed for learning:
- Python fundamentals,
- Data Structures and Algorithms in Python.

In this specific use case, instead of answering from the general internet or from the model`s pretraining, the tutor only answers using a small curated set of textbooks that you provide (here, open-licensed books like "A First Course on Data Structures in Python" by Donald R. Sheehy, MIT License © 2019 Don Sheehy).

If the answer is not present in the ingested sources/files, the tutor will explicitly mention:
``I do not have enough information in the provided documents.``

This makes it useful for focused self-study, classroom settings, and portfolio demonstrations of Retrieval-Augmented Generation (RAG).

---

## Features

- **Local / offline** inference using [Ollama](https://ollama.ai) and a lightweight model (Phi-3 3.8B).
- **Hugging Face Embeddings:** Uses `BAAI/bge-small-en-v1.5` for sentence-level semantic similarity. This balances retrieval quality and speed for educational documents.
- **RAG pipeline**:
   user question --> retrieve top chunks from PDF (Portable Document Format) sources via FAISS ( Facebook AI Similarity Search) --> pass both question + context to the model.
- **Citations**: answers include `[book.pdf, pX]` so learners can verify claims.
- **Refusal policy**: the tutor refuses to answer ("I do not have enough information in the provided documents."):
  - if the question is off-topic 
  - or if the information is not found in the PDFs.
- **Deterministic** responses (`temperature=0.0`) to reduce hallucinations.
- **Short/Long control**: you can ask "in 60 words", "detailed", "short", etc. Word budget is then enforced with soft trimming.

## How It Works?

1. **Ingest Phase (`ingest.py`)**
   - Reads all PDF files from `./data/`.
   - Splits each PDF into overlapping text chunks (configurable size).
   - Stores each chunk with metadata (source filename, page number)
   - Embeds chunks using a sentence embedding model (default: `BAAI/bge-small-en-v1.5`).
       **Why this model?**
     - `BAAI/bge-small-en-v1.5` is a compact, high-quality English text embedding model.  
     - It works offline and provides semantic similarity strong enough for retrieval tasks.  
     - `normalize_embeddings=True` ensures all vectors are normalized, improving FAISS search accuracy.
     - Builds a FAISS vector index from these embeddings and saves it to `./storage/faiss_index/`.
2. **Chat Phase (`tutor_cli.py`)**
   - You type a question.
   - The app retrieves the most relevant chunks from FAISS.
   - The local LLM (running in Ollama) generates an answer using **only from the retrieved context**.
   - The app trims the answer according to the requested word budget (e.g., "long", "short", "in 60 words") and displays citations where applicable.


## Setup
1.  **Prerequisites**
   You need:
      * Python 3.9+
      * [Ollama](https://ollama.ai) installed on your machine.
         * Ollama is what runs the local mode (Phi-3) instead of calling a paid API
      * Git
2.  **Clone/Create the repo locally**
   ```
   git clone https://github.com/karmesh825/Tutor_Bot---RAG-Assistant.git
   cd Tutor_Bot---RAG-Assistant
   ```
   If you're starting locally instead:
   Just make sure your folder matches the structure above (src/, data/, storage/ etc.)
   Then do:
      ```
      cd tutor
      git init
      ```


3.  **Create and activate a virtual environment**

   On Windows (PowerShell):
   ```
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
   On Linux / macOS:

   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   After activation, your terminal prompt should show (.venv) at the start.
   We do this to keep our install clean and avoids version fights between packages.
   
   
4.  **Install Python dependencies**
   From inside the project folder (with venv active):
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
6.  **Install and run Ollama**
   *1.* Install Ollama from [Ollama](https://ollama.ai) (one-time install).
   *2.* Start the Ollama service:
       ```
       ollama serve
       ```
       Leave that running in the background (or in another terminal).
   *3.* Pull the model used in the code (phi3):
       ```
       ollama oull phi3:3.8b
       ```
       This downloads the local LLM weights so tutor_cli.py can call it.
   If you use or change the model name in your code, you'll need to ollama pull <new_model_name> too.




7.  **Add your PDFs**
   Put your study PDFs into the data/ folder.
   * Only include content you are allowed to use(licensed product e.g, MIT, CC, etc.).
   * Do *not* commit such PDFs to GitHub of they are copyrighted.
     Your structure should look like:
        ```
        data/
           file_1.pdf
           file_2.pdf
           file_3.pdf
           ...
        ```


8.  **Build the vector index (ingestion step)**
   Run the ingest script once to create the FAISS index:
   ```
   python src/ingest.py
   ```
   This:
   * Reads each PDF in data/
   * Breaks it into text chunks
   * Embeds each chunk using `BAAI/bge-small-en-v1.5`.
     ```
     embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
     )
     ```
   * Saves a searchable FAISS index to storage/faiss_index/
   Why this matters:

   * The tutor will NOT work until this step runs successfully.
   * Re-run this anytime you add/remove PDFs.

   
9.  **Run the tutor CLI**

Now start the interactive assistant:
   ```
   python src/tutor_cli.py
   ```
You should see something like:
   ```
   ======================================================
   Hello! I am a DSA Python Tutor, how can I help you today?
   Ask a question (or "exit")
   You>
   ```
Try:
   ```
   You> explain dynamic programming
   Tutor> Dynamic Programming (DP) is an algorithmic technique that breaks down problems into simpler             subproblems and solves them just once while storing their solutions – a method known as memoization, or        solving each overlapping part only once by using tabulation[A First Course on Data Structures in Python.pdf]. This approach avoids the exponential time complexity of naive recursion for certain problems like    Fibonacci sequence calculation and can significantly reduce computation times to polynomial levels.

   ```
Try asking something off-topic:
   ```
   You> Do you know CSS or HTML?
   Tutor> I'm a DSA_Python_Tutor and my expertise is strictly within Python programming language as per your      request. I don't have knowledge of CSS or HTML, but if needed for web development projects involving these    languages, it would be best to consult resources specifically dedicated to them.
   ```
   The "refusal" is expected and correct.



-----------------------------------





