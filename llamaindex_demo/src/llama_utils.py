import os
from dotenv import load_dotenv

def get_llm():
    load_dotenv()
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "openai" or os.getenv("OPENAI_API_KEY"):
        from llama_index.llms.openai import OpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return OpenAI(model=model, temperature=0)
    else:
        # Use Ollama via LlamaIndex community wrapper
        from llama_index.llms.ollama import Ollama
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        return Ollama(model=model, temperature=0)