from __future__ import annotations
import pandas as pd

from . import config
from .utils import log, save_csv, load_csv

SUBREDDIT_LABEL = {
    "Ferrari": "Ferrari faithful",
    "formula1": "F1 fans",
    "electricvehicles": "EV enthusiasts",
    "cars": "Car enthusiasts",
    "wallstreetbets": "Traders / investors",
    "stocks": "Traders / investors", 
}

THEME_KEYWORDS = {
    "Apple/design": ["jony", "ive", "apple", "lovefrom"],
    "styling backlash": ["ugly", "hideous", "blob", "egg", "bubble"],
    "purist/identity": ["soul", "betrayal", "sellout", "heritage", "tradition"],
    "price/market": ["expensive", "overpriced", "shares", "margin", "stock"],
}

import re as _re

_SENT_COLS = ("sentiment_corrected", "transformer_label", "vader_label")

_BOT_AUTHORS = {"automoderator", "[deleted]", "[removed]", "none", "nan", "",
                "remindmebot", "visualmod", "ai-moderator"}
_BOT_SUFFIX = _re.compile(r"(?:^|[_-])bot$|modteam$")


def _is_human(author: str) -> bool:
    a = str(author).lower()
    return a not in _BOT_AUTHORS and not _BOT_SUFFIX.search(a)


def _load_docs() -> pd.DataFrame:
    """Best available per-doc table. Prefer the irony-corrected enriched table,
    but only if it is at least as fresh as documents_sentiment.csv (so a re-run of
    content_sentiment alone isn't shadowed by a stale enriched file)."""
    enriched = config.DATA_PROCESSED / "documents_enriched.csv"
    sentiment = config.DATA_PROCESSED / "documents_sentiment.csv"
    if enriched.exists() and (not sentiment.exists()
                              or enriched.stat().st_mtime >= sentiment.stat().st_mtime):
        df = load_csv("documents_enriched.csv")
        log.info("community sentiment: using documents_enriched.csv (%d docs)", len(df))
        return df
    if sentiment.exists():
        df = load_csv("documents_sentiment.csv")
        log.info("community sentiment: using documents_sentiment.csv (%d docs)", len(df))
        return df
    from .corpus import load_documents
    log.warning("no sentiment table found; using raw corpus (no labels).")
    return load_documents()


def _sentiment_col(df: pd.DataFrame) -> str | None:
    """First sentiment column that actually carries signal (>=2 distinct labels).
    Skips a transformer_label that is just an identical copy of vader_label
    (what the graceful lexicon fallback produces when transformers are absent)."""
    cats = {"positive", "neutral", "negative"}
    for c in _SENT_COLS:
        if c not in df.columns:
            continue
        vals = df[c].dropna()
        if vals.empty or vals[vals.isin(cats)].nunique() < 2:
            continue
        if c == "transformer_label" and "vader_label" in df.columns \
                and df[c].equals(df["vader_label"]):
            continue  # transformer lane fell back to VADER — no extra signal
        return c
    return next((c for c in _SENT_COLS if c in df.columns), None)


def _subreddit_by_author() -> dict[str, str]:
    """author -> their most frequent subreddit (for labeling when docs lack it)."""
    rows = []
    for name in ("reddit_submissions.csv", "reddit_comments.csv"):
        try:
            d = load_csv(name)
            if {"author", "subreddit"} <= set(d.columns):
                rows.append(d[["author", "subreddit"]])
        except FileNotFoundError:
            pass
    if not rows:
        return {}
    a = pd.concat(rows, ignore_index=True).dropna()
    return (a.groupby("author")["subreddit"]
              .agg(lambda s: s.value_counts().index[0]).to_dict())


def _top_keywords_per_community(docs: pd.DataFrame, comm_col: str,
                                top_k: int = 8) -> dict[int, list[str]]:
    """Distinctive terms per community via TF-IDF over per-community megadocs."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        log.warning("sklearn unavailable (%s); skipping keywords.", e)
        return {}
    grouped = docs.groupby(comm_col)["text"].apply(lambda s: " ".join(s.astype(str)))
    comms = grouped.index.tolist()
    if len(comms) < 2:
        return {}
    extra_stop = ["ferrari", "luce", "car", "cars", "ev", "electric", "just", "like",
                  "https", "com", "www", "amp", "don", "really", "people", "think"]
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop = list(ENGLISH_STOP_WORDS) + extra_stop
    except Exception:
        stop = extra_stop
    try:
        vec = TfidfVectorizer(stop_words=stop, min_df=1, max_df=0.5, max_features=4000,
                              ngram_range=(1, 1), token_pattern=r"[A-Za-z][A-Za-z]{2,}")
        X = vec.fit_transform(grouped.values)
    except ValueError as e:
        log.warning("community keyword TF-IDF skipped (%s); labels use subreddit only.", e)
        return {}
    vocab = vec.get_feature_names_out()
    out: dict[int, list[str]] = {}
    for i, c in enumerate(comms):
        row = X[i].toarray().ravel()
        out[int(c)] = [vocab[j] for j in row.argsort()[-top_k:][::-1] if row[j] > 0]
    return out


def _label(dominant_sub: str, sub_share: float, keywords: list[str],
           mix_threshold: float = 0.6) -> str:
    if dominant_sub and sub_share >= mix_threshold:
        base = SUBREDDIT_LABEL.get(dominant_sub, f"r/{dominant_sub}")
    else:
        base = "mixed / other"
    kw = set(keywords)
    scored = [(sum(w in kw for w in words), theme) for theme, words in THEME_KEYWORDS.items()]
    best_n, best_theme = max(scored, key=lambda x: x[0]) if scored else (0, None)
    return f"{base} — {best_theme}" if best_n > 0 else base


def build(min_docs: int = 20, comm_col: str = "community_louvain") -> pd.DataFrame:
    docs = _load_docs()
    if docs.empty:
        log.warning("empty corpus; run collectors + sentiment first.")
        return pd.DataFrame()
    try:
        comm = load_csv("nodes_communities.csv")
    except FileNotFoundError:
        log.warning("nodes_communities.csv missing; run build_graph + communities first.")
        return pd.DataFrame()

    sent_col = _sentiment_col(docs)
    if sent_col is None:
        log.warning("no sentiment column in docs; run content_sentiment first.")
        return pd.DataFrame()

    a2c = dict(zip(comm["node"].astype(str), comm[comm_col]))
    docs = docs.copy()
    docs["author"] = docs["author"].astype(str)
    docs["community"] = docs["author"].map(a2c)
    docs["is_human"] = docs["author"].map(_is_human)
    mapped = docs.dropna(subset=["community"])
    n_no_comm = int((docs["community"].isna() & docs["is_human"]).sum())
    matched = mapped[mapped["is_human"]].copy()
    matched["community"] = matched["community"].astype(int)
    log.info("Community sentiment: %d/%d docs mapped to a community "
             "(%d bot/system docs dropped, %d human docs have no community).",
             len(matched), len(docs), len(mapped) - len(matched), n_no_comm)

    if "subreddit" not in matched.columns or matched["subreddit"].fillna("").eq("").all():
        a2s = _subreddit_by_author()
        matched["subreddit"] = matched["author"].map(a2s).fillna("")

    keywords = _top_keywords_per_community(matched, "community")

    members = comm.groupby(comm_col)["node"].nunique()
    rows = []
    for c, g in matched.groupby("community"):
        if len(g) < min_docs:
            continue
        vc = g[sent_col].value_counts()
        pos, neu, neg = (int(vc.get(k, 0)) for k in ("positive", "neutral", "negative"))
        total = pos + neu + neg
        subs = g["subreddit"].replace("", pd.NA).dropna()
        dom_sub = subs.value_counts().index[0] if not subs.empty else ""
        sub_share = (subs.value_counts(normalize=True).iloc[0] if not subs.empty else 0.0)
        kws = keywords.get(int(c), [])
        rows.append({
            "community": int(c),
            "label": _label(dom_sub, float(sub_share), kws[:5]),
            "n_members": int(members.get(c, 0)),
            "n_docs": len(g),
            "dominant_subreddit": dom_sub,
            "subreddit_share": round(float(sub_share), 2),
            "positive": pos, "neutral": neu, "negative": neg,
            "negative_ratio": round(neg / total, 3) if total else 0.0,
            "mean_vader": round(float(g["vader_compound"].mean()), 3)
                          if "vader_compound" in g.columns else None,
            "top_keywords": ", ".join(kws[:6]),
        })
    out = (pd.DataFrame(rows)
           .sort_values("n_docs", ascending=False)
           .reset_index(drop=True))
    save_csv(out, "community_sentiment.csv")

    lbl = dict(zip(out["community"], out["label"]))
    matched["community_label"] = matched["community"].map(lbl)
    save_csv(matched[["doc_id", "author", "community", "community_label",
                      "subreddit", sent_col]].rename(columns={sent_col: "sentiment"}),
             "documents_communities.csv")

    camp_table(out)
    return out


def camp_table(per_community: pd.DataFrame) -> pd.DataFrame:
    """Roll up communities into discourse 'camps' — the headline view ('how do
    F1 fans / EV enthusiasts / traders feel?'). Camps are the BASE subreddit
    label (the ' — <theme>' refinement is a per-community detail and must NOT
    fragment the camp, or e.g. 'traders' would split into several rows)."""
    pc = per_community.copy()
    pc["camp"] = pc["label"].str.split(" — ").str[0]
    g = pc.groupby("camp").agg(
        n_communities=("community", "nunique"),
        n_members=("n_members", "sum"),
        n_docs=("n_docs", "sum"),
        positive=("positive", "sum"),
        neutral=("neutral", "sum"),
        negative=("negative", "sum"),
    ).reset_index().rename(columns={"camp": "label"})
    tot = g[["positive", "neutral", "negative"]].sum(axis=1).replace(0, 1)
    g["negative_ratio"] = (g["negative"] / tot).round(3)
    g["positive_ratio"] = (g["positive"] / tot).round(3)
    g = g.sort_values("n_docs", ascending=False).reset_index(drop=True)
    save_csv(g, "community_camps.csv")
    return g


def main() -> None:
    out = build()
    if out.empty:
        return
    cols = ["community", "label", "n_members", "n_docs", "dominant_subreddit",
            "negative_ratio", "top_keywords"]
    log.info("Per-community sentiment (top by volume):\n%s",
             out[cols].head(15).to_string(index=False))
    camps = load_csv("community_camps.csv")
    log.info("Discourse camps (communities rolled up by label):\n%s",
             camps[["label", "n_communities", "n_members", "n_docs",
                    "negative_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
