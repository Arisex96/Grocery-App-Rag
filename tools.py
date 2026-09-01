"""
tools.py
---------
Exposes clean, consolidated domain-based API wrappers mapping to Express.
Exposes schemas to Mistral AI via TOOL_SCHEMAS.
"""

import os
import json
import requests

EXPRESS_BASE = os.getenv("EXPRESS_API_URL", "http://localhost:4000/api")


# ─────────────────────────────────────────────
# Network Helpers
# ─────────────────────────────────────────────

def _post(path: str, body: dict) -> dict:
    """Helper to perform POST request to Express backend."""
    try:
        r = requests.post(f"{EXPRESS_BASE}{path}", json=body, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# Consolidated Tool Wrappers
# ─────────────────────────────────────────────

def search_products(query: str) -> dict:
    """Search for products in the catalog by name, category, or description."""
    return _post("/ai/products/search", {"query": query})


def get_cart(user_id: str) -> dict:
    """Fetch the contents and pricing total of the shopping cart for a user."""
    return _post("/ai/cart/get", {"userId": user_id})


def modify_cart(user_id: str, product_id: str, action: str, quantity: int = 1) -> dict:
    """
    Modify items in the shopping cart (add, remove, or clear).
    - action: "add" to add a product or increase quantity.
    - action: "remove" to reduce quantity or delete product when quantity drops to 0.
    - action: "clear" to remove all items from the cart.
    """
    return _post("/ai/cart/modify", {
        "userId": user_id,
        "productId": product_id,
        "action": action,
        "quantity": quantity
    })


def get_orders(user_id: str, query: str = "all") -> dict:
    """
    Retrieve order details for a user.
    - query: "latest" or "recent" to get their most recent order.
    - query: "all" or omit to list order history.
    - Specifying search terms (like a status term or order ID) lets you track active progress.
    """
    return _post("/ai/orders/get", {
        "userId": user_id,
        "query": query
    })


def search_deals(query: str = "") -> dict:
    """Find current product deals, bargains, and discount offers in the store (filtered optionally by a category keyword)."""
    return _post("/ai/deals/search", {"query": query})


def search_reviews(productId: str = None, query: str = None) -> dict:
    """
    Fetch reviews, ratings, and customer comments for a product.
    - productId: The unique product identifier.
    - query: Optional product name search query if ID isn't directly known.
    """
    return _post("/ai/reviews/search", {
        "productId": productId,
        "query": query
    })


# ─────────────────────────────────────────────
# Dispatcher mapping
# ─────────────────────────────────────────────

TOOL_FUNCTIONS = {
    "search_products":  lambda args: search_products(args.get("query", "")),
    "get_cart":         lambda args: get_cart(args["user_id"]),
    "modify_cart":      lambda args: modify_cart(
                            user_id=args["user_id"],
                            product_id=args.get("product_id"),
                            action=args["action"],
                            quantity=args.get("quantity", 1)
                        ),
    "get_orders":       lambda args: get_orders(args["user_id"], args.get("query", "all")),
    "search_deals":     lambda args: search_deals(args.get("query", "")),
    "search_reviews":   lambda args: search_reviews(productId=args.get("product_id"), query=args.get("query")),
}

def execute_tool(name: str, args: dict) -> str:
    """Executes a tool by name with parsed arguments and returns JSON string."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"Unknown tool: {name}"
    try:
        result = fn(args)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"}, indent=2)


# ─────────────────────────────────────────────
# Mistral Function Schemas
# ─────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Searches the catalog for products, categories, or tags. Returns pricing, inventory, and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search search query terms (e.g. 'fresh milk', 'organic fruits')"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_cart",
            "description": "Fetches items, total cost, and pricing info from the user's active shopping cart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The authenticated user's ID"}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "modify_cart",
            "description": "Adds, removes, or clears items in the user's active shopping cart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id":    {"type": "string", "description": "The authenticated user's ID"},
                    "product_id": {"type": "string", "description": "Unique product ID. Required unless clearing the entire cart."},
                    "action":     {"type": "string", "enum": ["add", "remove", "clear"], "description": "The mutation action"},
                    "quantity":   {"type": "integer", "description": "The quantity to add or remove (defaults to 1)"}
                },
                "required": ["user_id", "action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_orders",
            "description": "Lists recent order histories, tracking status, or ETAs for past purchases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "The authenticated user's ID"},
                    "query":    {"type": "string", "description": "Set to 'latest' for the single most recent order. Set to 'all' or search status terms/IDs to filter."}
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_deals",
            "description": "Fetches current discount sales, bargains, and clearance deals inside the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Optional keyword or category name (e.g. 'fruits', 'dairy')"}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_reviews",
            "description": "Queries feedback reviews, ratings, and customer comments for a product.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "The product's unique ID if directly known"},
                    "query":      {"type": "string", "description": "Optional product title keyword mapping if ID is unknown"}
                },
                "required": []
            }
        }
    }
]
