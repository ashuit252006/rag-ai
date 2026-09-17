import os

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()

app = Flask(__name__)

# -----------------------------
# Gemini API
# -----------------------------

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=API_KEY)

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")


# -----------------------------
# Embedding Model
# -----------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY
)


# -----------------------------
# ChromaDB
# -----------------------------

vectorstore = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="vectorstore"
)


# -----------------------------
# Home Page
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Chat API
# -----------------------------

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    question = data.get("question", "").strip()

    if not question:
        return jsonify({
            "answer": "Please enter a question."
        })

    try:

        # 1. Search PDF content
        results = vectorstore.similarity_search(
            question,
            k=3
        )

        if not results:
            return jsonify({
                "answer": "I could not find relevant information in the document."
            })

        # 2. Combine retrieved chunks
        context = "\n\n".join(
            doc.page_content
            for doc in results
        )

        # 3. RAG Prompt
        prompt = f"""
You are a helpful RAG document assistant.

Answer the user's question using ONLY the information
provided in the document context below.

If the answer is not available in the context,
say clearly:

"I could not find that information in the uploaded document."

Do not invent or assume information.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and concise answer.
"""

        # 4. Gemini generates answer
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        answer = response.text

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "answer": "Sorry, something went wrong while processing your question."
        }), 500


# -----------------------------
# Run Flask
# -----------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )