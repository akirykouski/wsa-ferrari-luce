"""LAB 5 — Sentiment, emotion, and Aspect-Based Sentiment Analysis.

Lexicon lane (Lane A, word-level): VADER + AFINN + NRC emotions.
Transformer lane (Lane B, BPE): cardiffnlp twitter-roberta sentiment.
ABSA: tag each doc with product aspects (look/sound/price/...) -> aspect x sentiment.

Transformer steps degrade gracefully: if transformers/torch aren't installed,
the lexicon results are still produced.

Run:  python -m src.content_sentiment
Outputs: documents_sentiment.csv , aspect_sentiment.csv , sentiment_timeline.csv
"""
from __future__ import annotations
import pandas as pd

from . import config
from .utils import log, save_csv, clean_for_transformer
from .corpus import load_documents

EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy",
            "sadness", "surprise", "trust"]


# ---------------------------------------------------------------- lexicon lane
def add_lexicon_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from afinn import Afinn
    vader = SentimentIntensityAnalyzer()
    afinn = Afinn()

    comp = df["text"].apply(lambda t: vader.polarity_scores(t)["compound"])
    df["vader_compound"] = comp
    df["vader_label"] = pd.cut(comp, [-1.01, -0.05, 0.05, 1.01],
                               labels=["negative", "neutral", "positive"])
    df["afinn_score"] = df["text"].apply(afinn.score)
    return df


def add_emotions(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from nrclex import NRCLex
    except Exception as e:  # noqa: BLE001
        log.warning("NRCLex unavailable (%s); skipping emotions.", e)
        return df
    def emo(t):
        freqs = NRCLex(t).affect_frequencies
        return [freqs.get(e, 0.0) for e in EMOTIONS]
    mat = df["text"].apply(emo).tolist()
    for i, e in enumerate(EMOTIONS):
        df[f"emo_{e}"] = [row[i] for row in mat]
    return df


# ------------------------------------------------------------ transformer lane
_PIPE_CACHE: dict = {}


def get_pipeline(task: str, model: str):
    key = (task, model)
    if key not in _PIPE_CACHE:
        from transformers import pipeline
        _PIPE_CACHE[key] = pipeline(task, model=model, truncation=True, max_length=512)
    return _PIPE_CACHE[key]


def add_transformer_sentiment(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    try:
        pipe = get_pipeline("sentiment-analysis", config.SENTIMENT_MODEL)
    except Exception as e:  # noqa: BLE001
        log.warning("transformer sentiment unavailable (%s); using VADER only.", e)
        df["transformer_label"] = df.get("vader_label")
        df["transformer_score"] = None
        return df
    texts = df["text"].apply(clean_for_transformer).tolist()
    labels, scores = [], []
    for i in range(0, len(texts), batch_size):
        for r in pipe(texts[i:i + batch_size]):
            labels.append(str(r["label"]).lower())
            scores.append(float(r["score"]))
        if i % (batch_size * 10) == 0:
            log.info("  transformer sentiment %d/%d", i, len(texts))
    df["transformer_label"] = labels
    df["transformer_score"] = scores
    return df


# --------------------------------------------------------------------- ABSA
def tag_aspects(text: str) -> list[str]:
    t = (text or "").lower()
    return [asp for asp, kws in config.ASPECTS.items() if any(k in t for k in kws)]


def aspect_table(df: pd.DataFrame, label_col: str = "transformer_label") -> pd.DataFrame:
    if label_col not in df.columns:
        label_col = "vader_label"
    df = df.copy()
    df["aspects"] = df["text"].apply(tag_aspects)
    exploded = df.explode("aspects").dropna(subset=["aspects"])
    tab = (exploded.groupby(["aspects", label_col]).size()
           .unstack(fill_value=0))
    for col in ("positive", "neutral", "negative"):
        if col not in tab.columns:
            tab[col] = 0
    tab["total"] = tab[["positive", "neutral", "negative"]].sum(axis=1)
    tab["negative_ratio"] = (tab["negative"] / tab["total"]).round(3)
    tab = tab.sort_values("total", ascending=False).reset_index()
    save_csv(tab, "aspect_sentiment.csv")
    return tab


def sentiment_timeline(df: pd.DataFrame) -> pd.DataFrame:
    d = df.dropna(subset=["created_at"]).copy()
    if d.empty:
        return pd.DataFrame()
    d = d.set_index("created_at")
    tl = d.resample("D").agg(
        n_docs=("doc_id", "count"),
        mean_compound=("vader_compound", "mean"),
    ).reset_index()
    save_csv(tl, "sentiment_timeline.csv")
    return tl


def main() -> None:
    df = load_documents()
    if df.empty:
        return
    df = add_lexicon_sentiment(df)
    df = add_emotions(df)
    df = add_transformer_sentiment(df)
    save_csv(df, "documents_sentiment.csv")
    aspect_table(df)
    sentiment_timeline(df)
    log.info("Sentiment label distribution:\n%s",
             df["transformer_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
