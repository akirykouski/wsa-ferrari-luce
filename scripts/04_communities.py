"""Stage 04 — community detection + modularity/assortativity (LAB 4)."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.communities import main

if __name__ == "__main__":
    main()
