"""One-off setup: download NLTK data and the spaCy English model."""
import subprocess
import sys

def main():
    import nltk
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        nltk.download(pkg, quiet=True)
    print("NLTK data downloaded.")
    subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
    print("Setup complete.")

if __name__ == "__main__":
    main()
