"""Configuration constants for the FEVER fact-verification agent."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

OLLAMA_URL = "http://127.0.0.1:11434"
CHAT_MODEL = "qwen2.5:7b-instruct"
EMBED_MODEL = "nomic-embed-text"

# Reproducible sampling of claims from the FEVER labelled_dev ("validation") split.
SEED = 42
N_CLAIMS_PER_LABEL = 20  # -> up to 60 claims total (fewer if a label runs short)
TOP_K_EVIDENCE = 5
MAX_SENTENCES_PER_PAGE = 60  # cap per Wikipedia page to keep the retrieval pool manageable

FEVER_LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

WIKI_PAGES_ZIP_URL = "https://fever.ai/download/fever/wiki-pages.zip"

WIKI_PAGES_CACHE = DATA_DIR / "wiki_pages_subset.json"
CLAIMS_CACHE = DATA_DIR / "claims_sample.json"
EMBED_CACHE = DATA_DIR / "embed_cache.json"
RESULTS_PATH = PROJECT_DIR / "results.json"
