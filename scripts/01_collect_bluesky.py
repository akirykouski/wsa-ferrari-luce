"""Stage 01 — collect Bluesky posts (LAB 2). Needs BLUESKY_* in .env."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.collect_bluesky import main

if __name__ == "__main__":
    main()
