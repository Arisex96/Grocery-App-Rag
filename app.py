"""
app.py
-------
FastAPI entrypoint for the Grocery AI Assistant.

On startup:
  1. Loads the RAG knowledge base (knowledge.txt) once.
  2. Creates a GroceryAgent that wraps the LLM + tools + RAG.

On POST /chat:
  - Accepts { "message": "...", "user_id": "..." }
  - Passes the message to the GroceryAgent.
  - Returns { "answer": "..." }
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import SimpleRAGSystem
from agent import GroceryAgent

# ── Environment ──────────────────────────────────────────────
load_dotenv()

api_key   = os.getenv("MISTRAL_API_KEY")
KB_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents", "knowledge.txt")

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

# ── Startup: build the agent once ────────────────────────────
agent: GroceryAgent | None = None

if not api_key:
    print("ERROR: MISTRAL_API_KEY is not set in .env")
else:
    try:
        print("Initialising RAG knowledge base...")
        rag_system = SimpleRAGSystem(api_key=api_key)
        rag_system.fit(KB_FILE)

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


@app.get("/health")
async def health():
    return {
        "status": "ok" if agent else "degraded",
        "kb_chunks": len(agent.rag.vector_db) if agent else 0,
        "tools_available": 12,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
