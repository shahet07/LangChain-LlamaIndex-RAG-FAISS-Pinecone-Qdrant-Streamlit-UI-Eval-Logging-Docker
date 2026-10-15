import os
from typing import Optional

# For OpenAI, we can estimate with tiktoken if installed.
def estimate_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        # Fallback heuristic: ~4 chars/token
        return max(1, len(text or "") // 4)

def detect_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"
