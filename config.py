"""
config.py - Central configuration for models, paths, thresholds, and prompt versions.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "twcs.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

for d in [PROCESSED_DATA_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Target Brand
TARGET_BRAND = "AmazonHelp"

# Threading Constraints
MAX_TURNS_PER_CASE = 4  # Truncate at 4 turns: Inbound -> Reply -> Customer Followup -> Final Reply

# Subsample Budgeting
RETRIEVAL_POOL_SIZE = 2500  # Historical knowledge base of resolved cases
EVAL_POOL_SIZE = 500        # Pool from which the golden eval set is sampled

# Free Model Endpoints & Defaults
# Supports Groq, Google AI Studio (Gemini OpenAI endpoint), or local Ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # 'groq' | 'gemini' | 'ollama'

MODELS = {
    "classifier": os.getenv("MODEL_CLASSIFIER", "openai/gpt-oss-20b"),
    "generator": os.getenv("MODEL_GENERATOR", "openai/gpt-oss-120b"),
    "judge": os.getenv("MODEL_JUDGE", "openai/gpt-oss-120b"),
}

API_KEYS = {
    "groq": os.environ.get("GROQ_API_KEY"),
    "gemini": os.environ.get("GEMINI_API_KEY"),
}

# Ensure keys are present if we aren't using local Ollama
if LLM_PROVIDER in ["groq", "gemini"] and not API_KEYS.get(LLM_PROVIDER):
    raise ValueError(f"Missing API key for provider: {LLM_PROVIDER}. Check your .env file.")

BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
}

# Embedding Model (Local CPU)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Prompt Versions
PROMPT_VERSIONS = {
    "classifier": "v1.0.0",
    "generator": "v1.0.0",
    "escalation": "v1.0.0",
    "judge": "v1.0.0",
}