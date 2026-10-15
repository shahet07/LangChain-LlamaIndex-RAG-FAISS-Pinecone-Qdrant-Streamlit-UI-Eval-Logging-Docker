import os
from dotenv import load_dotenv

def get_provider():
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider:
        return provider
    # Fallback: prefer OpenAI if key present, else Ollama
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"

def make_llm():
    provider = get_provider()
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        # Choose a sensible default small model; users can override via env OPENAI_MODEL
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0)
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        return ChatOllama(model=model, temperature=0)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
