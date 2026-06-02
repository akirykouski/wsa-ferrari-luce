"""Stage 05 — sentiment + emotion + aspect-based analysis (LAB 5)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.content_sentiment import main

if __name__ == "__main__":
    main()
