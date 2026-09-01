import os
import math
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
def load_documents(file_path: str) -> str:
    """Reads the knowledge base text file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge database file not found at: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
def split_text(text: str) -> list[str]:
    """
    Splits the text into meaningful paragraph-based chunks.
    Splits by headers or double newlines to keep topics grouped together.
    """
    # Split by double newlines to get distinct paragraphs/sections
    raw_chunks = text.split("\n\n")
    chunks = []
    
    current_chunk = []
    current_length = 0
    
    for section in raw_chunks:
        section = section.strip()
        if not section:
            continue
        
        # If adding this section exceeds 600 characters, save current chunk and start a new one
        if current_length + len(section) > 600 and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [section]
            current_length = len(section)
        else:
            current_chunk.append(section)
            current_length += len(section) + 2
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks
def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes the cosine similarity between two numeric vectors in pure Python."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)
class SimpleRAGSystem:
    def __init__(self, api_key: str):
        # Initialize Embeddings model and Chat model using langchain-mistralai
        self.embeddings_model = MistralAIEmbeddings(
            api_key=api_key,
            model="mistral-embed"
        )
        self.chat_model = ChatMistralAI(
            api_key=api_key,
            model="mistral-small-latest",
            temperature=0.2
        )
        self.vector_db = []  # In-memory storage list of dicts: {"text": str, "vector": list[float]}
    def fit(self, file_path: str):
        """Loads, splits, embeds, and stores the documents in memory."""
        print(f"Loading document: {file_path}")
        text = load_documents(file_path)
        
        print("Splitting text into chunks...")
        chunks = split_text(text)
        print(f"Generated {len(chunks)} chunks.")
        
        print("Generating embeddings for all chunks...")
        # Get embeddings list from Mistral API
        vectors = self.embeddings_model.embed_documents(chunks)
        
        # Store in our simple in-memory vector database
        self.vector_db = [
            {"text": chunk_t, "vector": vec}
            for chunk_t, vec in zip(chunks, vectors)
        ]
        print("Vector database initialization complete.")
    def retrieve(self, query: str, k: int = 2) -> list[dict]:
        """Runs similarity search using cosine similarity and returns top k chunks."""
        if not self.vector_db:
            print("Warning: Vector database is empty!")
            return []
            
        print(f"Computing query embedding for: '{query}'")
        query_vector = self.embeddings_model.embed_query(query)
        
        scored_chunks = []
        for item in self.vector_db:
            sim = cosine_similarity(query_vector, item["vector"])
            scored_chunks.append({"text": item["text"], "similarity": sim})
            
        # Sort by similarity score descending
        scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top k results
        return scored_chunks[:k]
    def answer_question(self, question: str) -> str:
        """Retrieves context and invokes LLM to generate an answer."""
        # 1. Retrieve the most relevant 2 chunks from the knowledge base
        relevant_chunks = self.retrieve(question, k=2)
        
        # Format the context block
        context_text = "\n---\n".join([item["text"] for item in relevant_chunks])
        
        print(f"Retrieved Context:\n{context_text}\n")
        
        # 2. Build the System prompt with retrieved context
        system_prompt = (
            "You are a helpful, friendly AI Grocery Assistant. "
            "Use the provided context from the knowledge base to answer the user's question.\n\n"
            f"Knowledge Base Context:\n{context_text}\n\n"
            "Rules:\n"
            "1. Answer the question base ONLY on the context information above.\n"
            "2. If you cannot find the answer in the context, say: 'I couldn't find details in my knowledge base. I can only assist with grocery items, storage, or recipes from my database.'\n"
            "3. Keep your answers concise, structured, and helpful."
        )
        
        # 3. Construct messages and send to ChatMistralAI
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        print("Invoking Mistral LLM...")
        response = self.chat_model.invoke(messages)
        return response.content
