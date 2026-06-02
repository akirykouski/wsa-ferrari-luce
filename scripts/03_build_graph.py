"""Stage 03 — build directed interaction graph + centralities (LAB 3)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.build_graph import main

if __name__ == "__main__":
    main()
