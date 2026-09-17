import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ GEMINI_API_KEY not found")
    exit()

# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)

# Connect to existing ChromaDB
vectorstore = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="vectorstore"
)

# User question
question = input("\n❓ Enter your question: ")

# Search relevant chunks
results = vectorstore.similarity_search(
    question,
    k=3
)

print("\n🔎 Relevant information found:\n")

for i, doc in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    print(doc.page_content)
    print()