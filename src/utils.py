"""Shared helpers: credentials, retry/backoff, text cleaning, I/O, language id.

Heavy / optional dependencies (nltk, langdetect) are imported lazily so that
collectors don't need the full ML stack, and so a missing package only breaks
the feature that uses it.
"""
from __future__ import annotations
import os
import re
import time
import logging
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from . import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("wsa")

# ----------------------------------------------------------------------------
# Credentials
# ----------------------------------------------------------------------------
def load_env() -> None:
    """Load .env into os.environ (no-op if python-dotenv is missing)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(config.BASE_DIR / ".env")
    except Exception:
        log.warning("python-dotenv not available; relying on existing env vars.")


def require_env(*keys: str) -> dict[str, str]:
    load_env()
    missing = [k for k in keys if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing credentials in environment/.env: {missing}. "
            f"Copy .env.example to .env and fill them in."
        )
    return {k: os.environ[k] for k in keys}


# ----------------------------------------------------------------------------
# Retry / backoff (LAB 2 pattern)
# ----------------------------------------------------------------------------
def with_retries(
    func: Callable,
    *args,
    max_attempts: int = 5,
    base: float = 2.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,),
    **kwargs,
):
    """Call func with exponential backoff on transient errors / rate limits."""
    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except exceptions as e:  # noqa: BLE001 - intentionally broad, re-raised below
            if attempt == max_attempts - 1:
                raise
            delay = min(base ** attempt, max_delay)
            log.warning("attempt %d/%d failed (%s); sleeping %.1fs",
                        attempt + 1, max_attempts, type(e).__name__, delay)
            time.sleep(delay)


# ----------------------------------------------------------------------------
# Text cleaning  (LAB 1 regex skills reused here instead of scraping news)
# ----------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[\w.]+")
HASHTAG_RE = re.compile(r"#(\w+)")
WS_RE = re.compile(r"\s+")


def clean_basic(text: str) -> str:
    """Light clean: strip URLs, collapse whitespace. Keeps #/@ as signal."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    return WS_RE.sub(" ", text).strip()


def clean_for_transformer(text: str) -> str:
    """cardiffnlp-recommended normalisation: @user / http placeholders.

    Used for the BPE / transformer lane (Lane B). Do NOT stopword/lemmatise here.
    """
    if not isinstance(text, str):
        return ""
    out = []
    for tok in text.split():
        if tok.startswith("@") and len(tok) > 1:
            out.append("@user")
        elif tok.startswith("http") or tok.startswith("www."):
            out.append("http")
        else:
            out.append(tok)
    return WS_RE.sub(" ", " ".join(out)).strip()


_NLTK_READY = False


def _ensure_nltk() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk
    for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass
    _NLTK_READY = True


def tokens_for_lexicon(text: str, lang: str = "english") -> list[str]:
    """Word-level lane (Lane A): TweetTokenizer + lowercase + stopwords + lemma.

    Feeds VADER/AFINN/NRC word stats, TF-IDF and word clouds.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    _ensure_nltk()
    from nltk.tokenize import TweetTokenizer
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    tok = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False)
    lemm = WordNetLemmatizer()
    try:
        stop = set(stopwords.words(lang))
    except Exception:
        stop = set()
    out = []
    for t in tok.tokenize(clean_basic(text)):
        if t in stop:
            continue
        if not re.search(r"[a-zA-Z#]", t):  # drop pure punctuation/numbers
            continue
        out.append(lemm.lemmatize(t))
    return out


def extract_hashtags(text: str) -> list[str]:
    return [h.lower() for h in HASHTAG_RE.findall(text or "")]


def is_bot(name) -> bool:
    """True for automated / official accounts (mod bots, karma/reminder bots,
    *-ModTeam). Used to exclude them from both the network and content layers."""
    if not isinstance(name, str):
        return False
    n = name.strip().lower()
    return bool(n) and (n in config.BOT_ACCOUNTS or n.endswith(config.BOT_SUFFIXES))


# ----------------------------------------------------------------------------
# Language detection (RQ8)
# ----------------------------------------------------------------------------
def detect_lang(text: str) -> str:
    if not isinstance(text, str) or len(text.strip()) < 3:
        return "unknown"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return "unknown"


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def save_csv(df: pd.DataFrame, name: str, processed: bool = True) -> Path:
    folder = config.DATA_PROCESSED if processed else config.DATA_RAW
    path = folder / name
    df.to_csv(path, index=False)
    log.info("saved %d rows -> %s", len(df), path)
    return path


def load_csv(name: str, processed: bool = True) -> pd.DataFrame:
    folder = config.DATA_PROCESSED if processed else config.DATA_RAW
    return pd.read_csv(folder / name)


def dedup(records: Iterable[dict], key: str) -> list[dict]:
    seen, out = set(), []
    for r in records:
        k = r.get(key)
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out
