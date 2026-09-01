"""
app.py
-------
FastAPI entrypoint for the Grocery AI Assistant.

On startup:
  1. Loads the RAG knowledge base (knowledge.txt) once.
  2. Fetches product catalog from Express backend and embeds static data.
  3. Creates a GroceryAgent that wraps the LLM + tools + RAG.

On POST /chat:
  - Accepts { "message": "...", "user_id": "..." }
  - Passes the message to the GroceryAgent.
  - Returns { "answer": "..." }

On POST /sync-products:
  - Re-fetches the product catalog from Express and updates embeddings.
"""

import os
import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import SimpleRAGSystem
from agent import GroceryAgent

# ── Environment ──────────────────────────────────────────────
load_dotenv()

api_key       = os.getenv("MISTRAL_API_KEY")
EXPRESS_BASE  = os.getenv("EXPRESS_API_URL", "http://localhost:4000/api")
KB_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents", "knowledge.txt")

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="Grocery AI Assistant",
    description="RAG + Function Calling chatbot for the Grocery-App."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helper: fetch product catalog from Express ────────────────
def _fetch_and_sync_products(rag_system: SimpleRAGSystem) -> int:
    """Fetches all products from Express backend and embeds static data."""
    try:
        r = http_requests.get(f"{EXPRESS_BASE}/ai/products/catalog", timeout=15)
        r.raise_for_status()
        products = r.json().get("products", [])
        count = rag_system.add_product_chunks(products)
        print(f"Product catalog synced: {count} products embedded.")
        return count
    except Exception as e:
        print(f"Could not sync products: {e}")
        return 0

# ── Startup: build the agent once ────────────────────────────
rag_system: SimpleRAGSystem | None = None
agent: GroceryAgent | None = None

if not api_key:
    print("ERROR: MISTRAL_API_KEY is not set in .env")
else:
    try:
        print("Initialising RAG knowledge base...")
        rag_system = SimpleRAGSystem(api_key=api_key)
        rag_system.fit(KB_FILE)

        # Try to sync product catalog from Express backend at startup
        _fetch_and_sync_products(rag_system)

        print("Creating GroceryAgent...")
        agent = GroceryAgent(api_key=api_key, rag_system=rag_system)
        print("Agent ready!")
    except Exception as e:
        print(f"Startup error: {e}")

# ── Request / Response models ─────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"   # frontend can pass the logged-in user's ID

class ChatResponse(BaseModel):
    answer: str

# ── Routes ───────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Empty message.")

    if agent is None:
        raise HTTPException(status_code=503, detail="AI service not ready (check MISTRAL_API_KEY).")

    try:
        answer = agent.chat(request.message, user_id=request.user_id)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")


@app.post("/sync-products")
async def sync_products():
    """Re-fetches the product catalog from Express and updates vector embeddings."""
    if rag_system is None:
        raise HTTPException(status_code=503, detail="RAG system not ready.")

    count = _fetch_and_sync_products(rag_system)
    kb_count = sum(1 for e in rag_system.vector_db if e.get("source") == "knowledge")
    return {
        "status": "ok",
        "products_synced": count,
        "knowledge_chunks": kb_count,
        "total_vectors": len(rag_system.vector_db),
    }


@app.get("/health")
async def health():
    product_count = sum(1 for e in agent.rag.vector_db if e.get("source") == "product") if agent else 0
    kb_count = sum(1 for e in agent.rag.vector_db if e.get("source") == "knowledge") if agent else 0
    return {
        "status": "ok" if agent else "degraded",
        "kb_chunks": kb_count,
        "product_chunks": product_count,
        "total_vectors": len(agent.rag.vector_db) if agent else 0,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
