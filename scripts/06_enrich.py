"""Stage 06 — sarcasm (RQ6), stance (RQ7), topic modelling, language (RQ8)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.content_enrich import main

if __name__ == "__main__":
    main()
