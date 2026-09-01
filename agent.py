"""
agent.py
---------
The main AI orchestration loop.
This file defines the GroceryAgent class which routes human messages:
  - If the user asks for store data (cart, order, deals, reviews): calls backend tools.
  - If the user asks for storage tips/recipes: retrieves local knowledge documents (RAG).
  - If the user just says hello or social chat: replies directly.
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_mistralai import ChatMistralAI

from rag import SimpleRAGSystem
from tools import TOOL_SCHEMAS, execute_tool

# Similarity score threshold (range 0.0 to 1.0).
# If the closest document similarity score is below this, we skip adding it to prompt context.
RAG_SIMILARITY_THRESHOLD = 0.65


def safe_print(msg: str):
    """Print helper to handle Windows terminals that might crash on certain emojis."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def remove_emojis(text: str) -> str:
    """Removes standard emojis to keep the chatbot's output clean and text-only."""
    if not text:
        return text
    cleaned = []
    for char in text:
        code_point = ord(char)
        # Miscellaneous Symbols (2600-26FF), Dingbats (2700-27BF), Pictographs/emoji blocks (1F000-1FAFF)
        if (0x2600 <= code_point <= 0x27BF) or (0x1F000 <= code_point <= 0x1FAFF):
            continue
        cleaned.append(char)
    return "".join(cleaned)


class GroceryAgent:
    """
    Orchestrates LLM messaging, multi-turn tool calling, and RAG retrieval.
    """
    def __init__(self, api_key: str, rag_system: SimpleRAGSystem):
        self.rag = rag_system

        # 1. Initialize the Mistral model
        base_llm = ChatMistralAI(
            api_key=api_key,
            model="mistral-small-latest",
            temperature=0.2
        )
        
        # 2. Bind the tools to the LLM so it knows which options are available
        self.llm = base_llm.bind_tools(TOOL_SCHEMAS)
        self.plain_llm = base_llm

    def chat(self, user_message: str, user_id: str = "anonymous") -> str:
        """
        Main chat entrypoint:
          1. Looks up information in the local knowledge base (RAG).
          2. Runs a loop (up to 8 turns) to let the LLM call tools consecutively.
        """
        safe_print(f"\n==================== USER REQUEST ====================")
        safe_print(f"User: {user_message}")
        safe_print(f"=======================================================")

        # Step 1: Look up local text documents for matching context
        rag_context = self._get_rag_context(user_message)

        # Step 2: Build starting messages list (System instruction + user message)
        messages = self._build_messages(user_message, user_id, rag_context)

        # Step 3: Run the multi-turn loop (up to 8 steps for complex tasks like
        #         search -> get product ID -> add to cart -> confirm)
        for turn in range(8):
            # Ask the AI model what to do next
            response = self.llm.invoke(messages)
            safe_print(f"\n[Turn {turn + 1}] AI Response: tool_calls={bool(response.tool_calls)}")

            # If the AI says it does not need to call any more tools, return its final text answer
            if not response.tool_calls:
                return remove_emojis(response.content)

            # Otherwise, the AI decided to call one or more tools.
            # We record the AI's decision/message in our history.
            messages.append(response)

            # Loop through each tool call requested by the AI
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call.get("args", {})

                # Ensure the current user's ID is always passed to the backend function
                if "user_id" not in tool_args or tool_args["user_id"] in ("", "anonymous", None):
                    tool_args["user_id"] = user_id

                safe_print(f"Executing: {tool_name}({json.dumps(tool_args)})")

                # Execute the tool (performs HTTP call to Express backend)
                result_str = execute_tool(tool_name, tool_args)
                safe_print(f"Result length: {len(result_str)} characters")

                # Feed the tool's output back to the message history so the AI can read it
                messages.append(
                    ToolMessage(content=result_str, tool_call_id=tool_call["id"])
                )

        return (
            "I wasn't able to complete that in time. "
            "Try being more specific, for example: "
            "'search for apples', 'show my cart', or 'what are my recent orders?'"
        )

    def _get_rag_context(self, query: str) -> str:
        """
        Checks if any paragraphs in the knowledge files match the user's message.
        Only returns matches if the similarity score is high enough.
        """
        if not self.rag or not self.rag.vector_db:
            return ""

        # Retrieve best 2 matching text chunks
        chunks = self.rag.retrieve(query, k=2)
        if not chunks:
            return ""

        # Check the similarity score of the best chunk
        best_score = chunks[0].get("similarity", 0)
        safe_print(f"RAG Best Match Similarity: {best_score:.3f}")

        # Skip context if similarity is too low (avoids feeding irrelevant text)
        if best_score < RAG_SIMILARITY_THRESHOLD:
            safe_print("RAG score too low - skipping knowledge base context.")
            return ""

        # Merge matching text chunks together
        context = "\n---\n".join(chunk["text"] for chunk in chunks)
        safe_print(f"RAG context injected ({len(context)} characters)")
        return context

    def _build_messages(self, user_message: str, user_id: str, rag_context: str) -> list:
        """
        Constructs the list of messages containing instructions and RAG context.
        """
        rag_section = ""
        if rag_context:
            rag_section = (
                f"\n\nKNOWLEDGE BASE (grocery/recipe info from local database):\n"
                f"{rag_context}\n"
                f"Use the knowledge base ONLY if it is directly relevant to the user's question."
            )

        system_content = (
            f"You are a helpful AI Grocery Assistant for the InstaCart app.\n"
            f"The current user's ID is: {user_id}\n\n"

            "=== TOOL-CALL RULES (MANDATORY) ===\n"
            "You have access to live store tools. Use them FIRST before saying anything.\n"
            "NEVER apologize or say you cannot check something without trying a tool first.\n\n"

            "Use tools for ALL of the following (no exceptions):\n"
            "- ANY question about whether a product exists, is available, or is in stock\n"
            "  -> call search_products(query=<product name>)\n"
            "- Viewing or managing the shopping cart\n"
            "  -> call get_cart or modify_cart\n"
            "- Adding, removing, or clearing items from the cart\n"
            "  -> call modify_cart(user_id, product_id, action, quantity)\n"
            "  -> If you don't have the product_id, call search_products first to find it\n"
            "- Checking order history or delivery status\n"
            "  -> call get_orders(user_id, query)\n"
            "- Finding deals, discounts, or sale items\n"
            "  -> call search_deals(query=<optional category>)\n"
            "- Reading product reviews or ratings\n"
            "  -> call search_reviews(product_id or query=<product name>)\n\n"

            "=== TOOL RESULT RULES ===\n"
            "- If search_products returns an empty list -> tell the user that product is not\n"
            "  currently available in the store. Do NOT say the system is down.\n"
            "- If a tool returns an error field -> retry once with simpler arguments before\n"
            "  telling the user there was a problem.\n"
            "- Always summarize tool results in plain, friendly English. Never show raw JSON.\n\n"

            "=== KNOWLEDGE BASE RULES ===\n"
            "Use the Knowledge Base ONLY for: storage tips, recipes, nutrition advice.\n"
            "Do NOT use the Knowledge Base for live store data (stock, prices, orders).\n\n"

            "=== GENERAL RULES ===\n"
            "- Never invent product IDs. Always call search_products first if you need one.\n"
            "- Never output raw JSON to the user.\n"
            "- Answer naturally and concisely.\n"
            f"{rag_section}"
        )

        return [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message)
        ]
