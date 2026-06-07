"""Unify Bluesky posts + Reddit submissions + Reddit comments into one corpus
that the content-analysis modules (sentiment, enrichments, NER) all share.

Columns: doc_id, source, author, created_at, text, lang
"""
from __future__ import annotations
import pandas as pd

from .utils import log, load_csv, clean_basic, detect_lang, is_bot


def _first_lang(langs: str) -> str:
    if isinstance(langs, str) and langs:
        return langs.split(",")[0]
    return ""


def load_documents() -> pd.DataFrame:
    frames = []

    try:
        b = load_csv("posts_bluesky.csv")
        if not b.empty:
            frames.append(pd.DataFrame({
                "doc_id": "bsky_" + b["uri"].astype(str),
                "source": "bluesky",
                "author": b["author_handle"],
                "created_at": b["created_at"],
                "text": b["text"].fillna(""),
                "lang": b["langs"].apply(_first_lang),
                "subreddit": "",
            }))
    except FileNotFoundError:
        pass

    try:
        s = load_csv("reddit_submissions.csv")
        if not s.empty:
            frames.append(pd.DataFrame({
                "doc_id": "rsub_" + s["id"].astype(str),
                "source": "reddit",
                "author": s["author"],
                "created_at": s["created_at"],
                "text": s["text"].fillna(""),
                "lang": "",
                "subreddit": s.get("subreddit", ""),
            }))
    except FileNotFoundError:
        pass

    try:
        c = load_csv("reddit_comments.csv")
        if not c.empty:
            frames.append(pd.DataFrame({
                "doc_id": "rcom_" + c["id"].astype(str),
                "source": "reddit",
                "author": c["author"],
                "created_at": c["created_at"],
                "text": c["body"].fillna(""),
                "lang": "",
                "subreddit": c.get("subreddit", ""),
            }))
    except FileNotFoundError:
        pass

    if not frames:
        log.warning("No collected data found; run the collectors first.")
        return pd.DataFrame(columns=["doc_id", "source", "author", "created_at", "text", "lang", "subreddit"])

    df = pd.concat(frames, ignore_index=True)
    df["subreddit"] = df["subreddit"].fillna("")
    df["text"] = df["text"].apply(clean_basic)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    n_before = len(df)
    df = df[~df["author"].map(is_bot)].reset_index(drop=True)
    if n_before - len(df):
        log.info("dropped %d bot-authored documents", n_before - len(df))
    mask = df["lang"].fillna("") == ""
    df.loc[mask, "lang"] = df.loc[mask, "text"].apply(detect_lang)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    log.info("Corpus: %d documents (%s)", len(df), df["source"].value_counts().to_dict())
    return df
