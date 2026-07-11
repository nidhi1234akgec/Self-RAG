from pathlib import Path
import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "deeplearning.pdf"

VECTOR_DB_PATH = BASE_DIR / "vectorstore"

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    api_key=MISTRAL_API_KEY,
)

# --------------------------------------------------
# Load existing vector database
# --------------------------------------------------

if VECTOR_DB_PATH.exists():

    print("Loading existing FAISS index...")

    vector_store = FAISS.load_local(
        str(VECTOR_DB_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

# --------------------------------------------------
# Otherwise create it
# --------------------------------------------------

else:

    print("Creating vector database...")

    docs = PyPDFLoader(str(PDF_PATH)).load()

    chunks = RecursiveCharacterTextSplitter(

        chunk_size=600,

        chunk_overlap=150,

    ).split_documents(docs)

    vector_store = FAISS.from_documents(

        chunks,

        embeddings,

    )

    VECTOR_DB_PATH.mkdir(exist_ok=True)

    vector_store.save_local(str(VECTOR_DB_PATH))

    print("Vector database saved successfully!")

retriever = vector_store.as_retriever(

    search_kwargs={"k": 4}

)