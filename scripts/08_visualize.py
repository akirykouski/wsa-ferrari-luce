"""Stage 08 — render all figures to figures/."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from src.viz import main

if __name__ == "__main__":
    main()
