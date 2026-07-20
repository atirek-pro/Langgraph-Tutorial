import os
from pathlib import Path

# Set Hugging Face cache directories before importing HuggingFaceEmbeddings.
# Some HF-backed libraries read these environment variables during import,
# so this ensures the downloaded embedding model and cache stay inside the
# project-local `Langsmith/models/embedding_model` folder instead of the
# default user/system Hugging Face cache location.
BASE_DIR = Path(__file__).resolve().parent
EMBEDDING_CACHE_DIR = BASE_DIR / "models" / "embedding_model"
os.environ["HF_HOME"] = str(BASE_DIR / "models")
os.environ["TRANSFORMERS_CACHE"] = str(EMBEDDING_CACHE_DIR)

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()  # expects API_KEY in .env

os.environ["LANGCHAIN_PROJECT"] = "RAG_PROJECT_V1"

BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "data" / "islr.pdf"  # <-- change to your PDF filename if needed
FAISS_DIR = BASE_DIR / "faiss_vector_index"
EMBEDDING_CACHE_DIR = BASE_DIR / "models" / "embedding_model"

FAISS_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(BASE_DIR / "models")
os.environ["TRANSFORMERS_CACHE"] = str(EMBEDDING_CACHE_DIR)

# 1) Load PDF
loader = PyPDFLoader(str(PDF_PATH))
docs = loader.load()  # one Document per page

# 2) Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
splits = splitter.split_documents(docs)

# 3) Embed + index
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vs = FAISS.from_documents(splits, emb)
vs.save_local(str(FAISS_DIR))
retriever = FAISS.load_local(
    str(FAISS_DIR),
    emb,
    allow_dangerous_deserialization=True,
).as_retriever(search_type="similarity", search_kwargs={"k": 4})

# 4) Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know."),
    ("human", "Question: {question}\n\nContext:\n{context}")
])

# 5) Chain
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)

def format_docs(docs): return "\n\n".join(d.page_content for d in docs)

parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()

# 6) Ask questions
print("PDF RAG ready. Ask a question (or Ctrl+C to exit).")
q = input("\nQ: ")
ans = chain.invoke(q.strip())
print("\nA:", ans)
