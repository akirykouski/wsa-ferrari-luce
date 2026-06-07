from __future__ import annotations
import pandas as pd

from . import config
from .utils import log, save_csv, load_csv, clean_for_transformer
from .corpus import load_documents
from .content_sentiment import get_pipeline

STANCE_MAX = 2000
TARGET_PHRASES = {"luce": "the Ferrari Luce", "ev_transition": "electric vehicles"}
STANCE_LABELS = {"in favor of": "favor", "against": "against", "neutral about": "neutral"}


def _load_base() -> pd.DataFrame:
    try:
        df = load_csv("documents_sentiment.csv")
        log.info("Loaded sentiment-scored corpus (%d docs).", len(df))
        return df
    except FileNotFoundError:
        log.warning("documents_sentiment.csv missing; using raw corpus (no correction).")
        return load_documents()


def add_irony(df: pd.DataFrame, batch_size: int = 32) -> pd.DataFrame:
    try:
        pipe = get_pipeline("text-classification", config.IRONY_MODEL)
    except Exception as e:
        log.warning("irony model unavailable (%s); skipping irony detection.", e)
        df["irony_flag"] = False
        return df
    texts = df["text"].apply(clean_for_transformer).tolist()
    flags, scores = [], []
    for i in range(0, len(texts), batch_size):
        for r in pipe(texts[i:i + batch_size]):
            lab = str(r["label"]).lower()
            is_irony = ("iron" in lab and "non" not in lab) or lab in ("label_1", "1")
            flags.append(bool(is_irony))
            scores.append(float(r["score"]))
    df["irony_flag"] = flags
    df["irony_score"] = scores
    if "transformer_label" in df.columns:
        def correct(row):
            lab = row.get("transformer_label")
            if row["irony_flag"] and lab == "positive":
                return "negative"
            return lab
        df["sentiment_corrected"] = df.apply(correct, axis=1)
    log.info("Irony prevalence: %.1f%%", 100 * df["irony_flag"].mean())
    return df


def add_stance(df: pd.DataFrame) -> pd.DataFrame:
    try:
        zsc = get_pipeline("zero-shot-classification", config.STANCE_NLI_MODEL)
    except Exception as e:
        log.warning("stance NLI model unavailable (%s); skipping stance.", e)
        return df
    n = min(len(df), STANCE_MAX)
    if n < len(df):
        log.warning("stance capped at %d/%d docs for runtime.", n, len(df))
    sub = df.head(n)
    texts = sub["text"].apply(clean_for_transformer).tolist()
    label_phrases = list(STANCE_LABELS.keys())
    for target, phrase in TARGET_PHRASES.items():
        cands = [f"{lp} {phrase}" for lp in label_phrases]
        out = []
        for t in texts:
            res = zsc(t, candidate_labels=cands, multi_label=False)
            top = res["labels"][0]
            prefix = next(lp for lp in label_phrases if top.startswith(lp))
            out.append(STANCE_LABELS[prefix])
        df.loc[sub.index, f"stance_{target}"] = out
        log.info("stance(%s): %s", target,
                 pd.Series(out).value_counts().to_dict())
    return df


def topic_model(df: pd.DataFrame, n_topics: int = 8) -> pd.DataFrame:
    texts = df["text"].tolist()
    try:
        from bertopic import BERTopic
        tm = BERTopic(language="multilingual", verbose=False)
        topics, _ = tm.fit_transform(texts)
        df["topic"] = topics
        info = tm.get_topic_info()[["Topic", "Count", "Name"]]
        save_csv(info.rename(columns=str.lower), "topics_keywords.csv")
        log.info("BERTopic found %d topics.", len(info))
        return df
    except Exception as e:
        log.warning("BERTopic unavailable (%s); falling back to sklearn LDA.", e)
    try:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        vec = CountVectorizer(max_df=0.9, min_df=2, stop_words="english")
        X = vec.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        W = lda.fit_transform(X)
        vocab = vec.get_feature_names_out()
        rows = [{"topic": k,
                 "keywords": ", ".join(vocab[i] for i in comp.argsort()[-10:][::-1])}
                for k, comp in enumerate(lda.components_)]
        save_csv(pd.DataFrame(rows), "topics_keywords.csv")
        df["topic"] = W.argmax(axis=1)
        log.info("LDA found %d topics.", n_topics)
    except Exception as e:
        log.warning("topic modelling failed (%s).", e)
    return df


def language_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    label_col = "sentiment_corrected" if "sentiment_corrected" in df.columns else (
        "transformer_label" if "transformer_label" in df.columns else "vader_label")
    if label_col not in df.columns:
        log.warning("no sentiment column for language segmentation.")
        return pd.DataFrame()
    d = df.copy()
    d["lang_group"] = d["lang"].where(d["lang"].isin(config.LANGUAGES), "other")
    tab = (d.groupby(["lang_group", label_col]).size().unstack(fill_value=0))
    tab["total"] = tab.sum(axis=1)
    if "negative" in tab.columns:
        tab["negative_ratio"] = (tab["negative"] / tab["total"]).round(3)
    tab = tab.reset_index()
    save_csv(tab, "language_sentiment.csv")
    log.info("Language x sentiment:\n%s", tab.to_string(index=False))
    return tab


def main() -> None:
    df = _load_base()
    if df.empty:
        return
    df = add_irony(df)
    df = add_stance(df)
    df = topic_model(df)
    save_csv(df, "documents_enriched.csv")
    language_segmentation(df)


if __name__ == "__main__":
    main()
