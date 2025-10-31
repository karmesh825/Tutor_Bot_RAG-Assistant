import re, os
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate

#path to faiss index folder 
DB_DIR = "your_folder_name/storage/faiss_index"

#control knobs
DEFAULT_MAX_WORDS = 80 # default brevity(use of brief expressions/word budget)
MAX_TOKENS = 128       # hard cap on tokens the model can emit
RETRIEVER_K = 3   # num of chunks, fewer chunks means tighter answers
TEMPERATURE = 0.0 # low means more crisp/deterministic

#build stack
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={"normalize_embeddings": True})
vectordb = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVER_K})

llm = ChatOllama(model="phi3:3.8b", temperature= TEMPERATURE,num_predict=MAX_TOKENS)

SYSTEM = (
    "You are a concise DSA_Python_Tutor. Be on-topic. "
    "Do NOT rely on prior knowledge, or external training."
    "If the question cannot be answered, respond exactly with:'I don't have enough information in the provided documents.'"
    "Never discuss topics outside Python or the given context."
    "Do NOT repeat or restate the user's question or any text prompt."
    "Cite briefly like [doc, pX] when possible, also if the answer is adopted from the refernce [doc]." 
    "No prefaces like 'AI:', 'User:'."
    "Prefer clarity over brevity; if the word limit is tight, complete the thought in full sentences."
    "Do not repeat the user's question. Do not guess. "
    "If context is empty or irrelevant, just use the refusal message."
    
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("system","Reference material(do not quote labels): \n{context}"),
    ("human", "{question}\n\n Write the answer in <= {max_words} words.")
    #"If not found, reply exactly: 'I do not have enough information in the provided documents.'\n\n"
])


#=====================================
#helper for picking reliable metadata
def compute_display_page(meta: dict) -> str:
    if not meta:
        return "?"
    #in case you add roman numerals later
    if "page_label" in meta and meta["page_label"]:
        return str(meta["page_label"])
    #0-based physical index
    if "page_index0" in meta and meta["page_index0"] is not None:
        try:
            return str(int(meta["page_index0"]) + 1)
        except Exception:
            pass
    #Try load 'page'
    if "page" in meta and meta["page"] is not None:
        try:
            return str(int(meta["page"]) + 1)
        except Exception:
            pass
    return "?"


def format_docs(docs):
    blocks = []
    for d in docs:
        meta = d.metadata or {}

        title = os.path.basename(meta.get("source", "doc"))

        # p = meta.get("page")
        page_display = compute_display_page(meta)

        if page_display != "?":
            # human_page = int(p)-1
            cite = f"[{title}, p{page_display}]"
        else:
            #human_page = int(p)
            cite = f"[{title}]"

        blocks.append(f"{d.page_content}\n{cite}")

    return "\n\n---\n\n".join(blocks)



    
def parse_word_limit(q: str, default: int = DEFAULT_MAX_WORDS) -> int:
    """
    Extract a word budget from the query if present:
    e.g., 'in 60 words', 'under 100 words', 'less than 50 words', 'brief'/'short'/'long'/'detailed'
    """
    q_low = q.lower()

    #explicit numeric/phrase limits
    m = re.search(r"(?:in|under|less than|at most|<=)\s*(\d+)\s*words?", q_low)
    if m:
        return max(10, int(m.group(1)))


    if any(k in q_low for k in ["one line", "1 line"]):
        return 20
    if "brief" in q_low or "short" in q_low or "concise" in q_low:
        return 60
    if "detailed" in q_low or "long" in q_low or "elaborate" in q_low:
        return 180

    return default

GREETINGS = {"Hi!","Hi there!", "Hello!", "Hey",
            "hi","hi there", "hello", "hey",
            "hi!","hi there!", "hello!", "hey!"}

def is_greeting(q: str) -> bool:
    return q.strip().lower() in GREETINGS



def soft_sentence_trim(text:str, max_words:int)->str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    hard_cap = max_words + 15 #basically adding maximum num of words to finish a sentence.
    snippet = " ".join(words[:min(len(words),hard_cap)])

    #cutting the sentence terminators
    cut = max(snippet.rfind(". "),snippet.rfind("! "),snippet.rfind("? "))
    if cut != -1:
        return snippet[:cut+1].strip()

print('======================================================')
print('Hello! I am a DSA Python Tutor, how can I help you today? ')
print('Ask a question (or "exit")')

while True:
    q = input("You> ").strip()
    if q.lower() in {"exit", "quit"}:
        break

    #handle greetings without invoking the big model(without retrieval/LLM)
    if is_greeting(q):
        print("Tutor> Hi! What Python topic should we discuss? \n")
        continue
    
    #fewer docs reduces verbosity
    docs = retriever.invoke(q)
    ctx = format_docs(docs)

    max_words = parse_word_limit(q)

    msg = prompt.format_messages(
        question=q,
        context=ctx if ctx else "(no relevant snippets found)",
        max_words=max_words
    )

    ans = llm.invoke(msg)
    #As an extra guard we truncate to the word budget (in case the model overflows slightly)
    text = soft_sentence_trim(ans.content,max_words)
    

    print(f"Tutor> {text}\n")
