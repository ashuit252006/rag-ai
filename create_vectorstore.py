import os
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ GEMINI_API_KEY not found in .env")
    exit()

# -----------------------------
# 1. Find PDF
# -----------------------------

PDF_FOLDER = Path("documents")
pdf_files = list(PDF_FOLDER.glob("*.pdf"))

if not pdf_files:
    print("❌ No PDF found inside documents folder")
    exit()

pdf_path = pdf_files[0]

print(f"📄 PDF: {pdf_path.name}")


# -----------------------------
# 2. Extract text
# -----------------------------

reader = PdfReader(str(pdf_path))

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print(f"📖 Pages: {len(reader.pages)}")
print(f"📝 Characters extracted: {len(text)}")


# -----------------------------
# 3. Split into chunks
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_text(text)

print(f"🧩 Total chunks: {len(chunks)}")


# -----------------------------
# 4. Create Embedding Model
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

print("🧠 Embedding model initialized")


# -----------------------------
# 5. Store embeddings in ChromaDB
# -----------------------------

vectorstore = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="vectorstore"
)

# Add documents
vectorstore.add_texts(chunks)

print("✅ Embeddings created successfully")
print("✅ Documents stored in ChromaDB")
print("📁 Vector database location: vectorstore/")