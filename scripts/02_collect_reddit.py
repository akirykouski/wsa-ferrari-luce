"""Stage 02 — collect Reddit submissions + comments (LAB 2). Needs REDDIT_* in .env."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.collect_reddit import main

if __name__ == "__main__":
    main()
