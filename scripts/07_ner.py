"""Stage 07 — NER + entity-level sentiment (LAB 6)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.ner_entities import main

if __name__ == "__main__":
    main()
