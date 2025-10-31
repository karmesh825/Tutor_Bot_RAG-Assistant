import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_DIR = "D:/AI_Topics/python_tutor/data"
DB_DIR = "D:/AI_Topics/python_tutor/storage/faiss_index"
CHUNK_SIZE = 128
CHUNK_OVERLAP = 20

def load_pdfs(path):
    # docs = []
    # for f in os.listdir(path):
    #     if f.lower().endswith(".pdf"):
    #         loader = PyPDFLoader(os.path.join(path, f))
    #         docs.extend(loader.load())
    # return docs
    docs = []
    for f in os.listdir(path):
        if f.lower().endswith(".pdf"):
            full_path = os.path.join(path, f)
            loader = PyPDFLoader(full_path)
            pages = loader.load()  #one Document per page

            for p_idx, d in enumerate(pages):
                #make sure metadata exists
                if not hasattr(d, "metadata") or d.metadata is None:
                    d.metadata = {}

                #0-based index
                d.metadata["source_name"] = os.path.basename(full_path)
                d.metadata["page_index0"] = p_idx  # 0-based physical page index

            docs.extend(pages)
    return docs


def split_keep_meta(page_docs):
    
    #Split each page into smaller chunks but keep page metadata so later we can cite [book.pdf, page_number].
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    out_chunks = []
    for page_doc in page_docs:
        sub_chunks = splitter.split_documents([page_doc])
        for c in sub_chunks:

            if not hasattr(c, "metadata") or c.metadata is None:
                c.metadata = {}

            # copy forward metadata safely using .get()
            src_name = page_doc.metadata.get("source_name", "unknown.pdf")
            idx0 = page_doc.metadata.get("page_index0", None)
            raw_page = page_doc.metadata.get("page", idx0)

            c.metadata["source_name"] = src_name
            c.metadata["page_index0"] = idx0
            c.metadata["page"] = raw_page


            #note: PyPDFLoader also puts "page" in metadata; we keep it too
            #that gives us multiple options for referencing later
        out_chunks.extend(sub_chunks)
    return out_chunks
if __name__ == "__main__":
    load_dotenv()
    os.makedirs(DB_DIR, exist_ok=True)

    print("Loading PDFs...")
    raw_docs = load_pdfs(DATA_DIR)

    print(f"Loaded {len(raw_docs)} pages. Splitting...")
    chunks = split_keep_meta(raw_docs)
    print(f"Got {len(chunks)} chunks.")

    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    print(f"Using HF embeddings: {model_name}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Building FAISS index...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(DB_DIR)
    print(f"Saved index to {DB_DIR}")
