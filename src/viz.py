"""Visualization helpers. Each function reads a processed CSV and writes a figure
to figures/. Functions skip silently if their input isn't there yet, so you can
run this after any subset of the pipeline.

Run:  python -m src.viz
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:  # seaborn gives the charts a cleaner default look (optional)
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
except Exception:  # noqa: BLE001
    pass

from . import config
from .utils import log, load_csv

SENT_COLORS = {"negative": "#d62728", "neutral": "#7f7f7f", "positive": "#2ca02c"}


def _save(fig, name: str):
    path = config.FIGURES / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("figure -> %s", path)


def _try(fn):
    try:
        fn()
    except FileNotFoundError:
        log.info("skip %s (input missing)", fn.__name__)
    except Exception as e:  # noqa: BLE001
        log.warning("%s failed: %s", fn.__name__, e)


# ------------------------------------------------------------------- content
def sentiment_distribution():
    df = load_csv("documents_enriched.csv") if (config.DATA_PROCESSED / "documents_enriched.csv").exists() else load_csv("documents_sentiment.csv")
    col = next((c for c in ("sentiment_corrected", "transformer_label", "vader_label") if c in df.columns), None)
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    counts.reindex(["negative", "neutral", "positive"]).plot(
        kind="bar", ax=ax, color=[SENT_COLORS.get(i, "#888") for i in ["negative", "neutral", "positive"]])
    ax.set_title(f"Sentiment distribution ({col})")
    ax.set_ylabel("documents")
    _save(fig, "sentiment_distribution.png")


def emotions():
    df = load_csv("documents_sentiment.csv")
    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    if not emo_cols:
        return
    means = df[emo_cols].mean().sort_values(ascending=False)
    means.index = [c.replace("emo_", "") for c in means.index]
    if means.sum() == 0:  # all-zero -> NRCLex didn't populate; don't emit a blank chart
        log.warning("emotions: all emo_ columns are zero; skipping (check add_emotions/NRCLex).")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    means.plot(kind="bar", ax=ax, color="#9467bd")
    ax.set_title("Mean NRC emotion intensity")
    ax.set_ylabel("mean affect frequency")
    ax.tick_params(axis="x", rotation=45)
    _save(fig, "emotions.png")


def aspect_sentiment():
    tab = load_csv("aspect_sentiment.csv").set_index("aspects")
    parts = [c for c in ("negative", "neutral", "positive") if c in tab.columns]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    tab[parts].plot(kind="bar", stacked=True, ax=ax,
                    color=[SENT_COLORS[p] for p in parts])
    ax.set_title("Aspect-based sentiment (Ferrari Luce)")
    ax.set_ylabel("documents")
    _save(fig, "aspect_sentiment_stacked.png")

    if "negative_ratio" in tab.columns:
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        tab["total"].plot(kind="pie", ax=ax2, autopct="%1.0f%%",
                          colors=plt.cm.Set3.colors)
        ax2.set_ylabel("")
        ax2.set_title("Share of discussion per aspect")
        _save(fig2, "aspect_share_pie.png")


def timeline():
    tl = load_csv("sentiment_timeline.csv")
    tl["created_at"] = pd.to_datetime(tl["created_at"], errors="coerce", utc=True)
    tl = tl.dropna(subset=["created_at"]).set_index("created_at")
    # Daily bins are too sparse (most days are empty) -> aggregate to weekly.
    # Weekly mean sentiment is the post-weighted mean: daily mean * count = daily
    # sum, so sum(mean*n)/sum(n) recovers the true weekly mean.
    tl["sum_compound"] = tl["mean_compound"].fillna(0) * tl["n_docs"]
    wk = tl.resample("W").agg(n_docs=("n_docs", "sum"),
                              sum_compound=("sum_compound", "sum"))
    wk["mean_compound"] = (wk["sum_compound"] / wk["n_docs"]).where(wk["n_docs"] > 0)
    active = wk[wk["n_docs"] > 0]  # line only across weeks that actually have posts

    fig, ax1 = plt.subplots(figsize=(11, 5))
    bars = ax1.bar(wk.index, wk["n_docs"], width=5, color="#aec7e8",
                   edgecolor="#7fa8d0", label="posts per week")
    ax1.set_ylabel("posts per week", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    line, = ax2.plot(active.index, active["mean_compound"], color="#d62728",
                     marker="o", ms=4, lw=1.8, label="mean VADER compound")
    ax2.axhline(0, color="grey", lw=0.6, ls="--")
    ax2.set_ylabel("mean VADER compound", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.set_ylim(-1, 1)

    # Event markers (single shared legend entry).
    evt_handle = None
    for date, lbl in config.EVENT_DATES:
        try:
            evt_handle = ax1.axvline(pd.Timestamp(date, tz="UTC"),
                                     color="black", lw=0.9, ls=":")
        except Exception:
            pass

    handles = [bars, line] + ([evt_handle] if evt_handle is not None else [])
    labels = ["posts per week", "mean VADER compound"] + (["key events"] if evt_handle else [])
    ax1.legend(handles, labels, loc="upper left", framealpha=0.9)
    ax1.set_title("Volume & sentiment over time (weekly)")
    _save(fig, "sentiment_timeline.png")


def wordclouds():
    """One cloud per sentiment, sized by how *distinctive* each term is to that
    class rather than by raw frequency.

    Raw-frequency clouds are dominated by words common to every class ("people",
    "think", "design"...), which carry no contrast. Instead we TF-IDF the whole
    corpus (uni- + bi-grams) so corpus-wide words are downweighted, drop near-
    universal terms via max_df, then rank each class by its mean TF-IDF weight.
    """
    try:
        from wordcloud import WordCloud, STOPWORDS
    except Exception:
        return
    src = "documents_enriched.csv" if (config.DATA_PROCESSED / "documents_enriched.csv").exists() else "documents_sentiment.csv"
    df = load_csv(src)
    col = next((c for c in ("sentiment_corrected", "transformer_label", "vader_label") if c in df.columns), None)
    if col is None:
        return
    df = df[["text", col]].dropna()
    df["text"] = df["text"].astype(str)
    sentiments = ("negative", "neutral", "positive")
    stop = set(STOPWORDS) | config.WORDCLOUD_STOPWORDS

    try:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
    except Exception:  # noqa: BLE001 - fall back to plain frequency clouds below
        TfidfVectorizer = None

    if TfidfVectorizer is not None:
        vec = TfidfVectorizer(stop_words=list(stop | set(ENGLISH_STOP_WORDS)),
                              ngram_range=(1, 2), min_df=5, max_df=0.4,
                              sublinear_tf=True)
        try:
            X = vec.fit_transform(df["text"])
        except ValueError:
            X = None
        if X is not None:
            vocab = vec.get_feature_names_out()
            for sentiment in sentiments:
                mask = (df[col] == sentiment).to_numpy()
                if mask.sum() < 5:
                    continue
                weights = np.asarray(X[mask].mean(axis=0)).ravel()
                top = weights.argsort()[::-1][:120]
                freqs = {vocab[i]: float(weights[i]) for i in top if weights[i] > 0}
                if not freqs:
                    continue
                wc = WordCloud(width=800, height=400, background_color="white",
                               collocations=False, prefer_horizontal=0.9
                               ).generate_from_frequencies(freqs)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.imshow(wc); ax.axis("off")
                ax.set_title(f"{sentiment} posts — distinctive terms")
                _save(fig, f"wordcloud_{sentiment}.png")
            return

    # Fallback (sklearn unavailable): plain frequency clouds.
    for sentiment in sentiments:
        text = " ".join(df.loc[df[col] == sentiment, "text"])
        if len(text) < 50:
            continue
        wc = WordCloud(width=800, height=400, background_color="white",
                       stopwords=stop, collocations=False).generate(text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc); ax.axis("off"); ax.set_title(f"{sentiment} posts")
        _save(fig, f"wordcloud_{sentiment}.png")


# ------------------------------------------------------------------- network
def centrality_top(n: int = 15):
    df = load_csv("nodes_centrality.csv").head(n)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(df["node"][::-1], df["pagerank"][::-1], color="#ff7f0e")
    ax.set_title(f"Top {n} accounts by PageRank")
    _save(fig, "centrality_top.png")


def entity_frequency(n: int = 20):
    df = load_csv("entity_frequency.csv").head(n)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(df["entity"][::-1], df["count"][::-1], color="#17becf")
    ax.set_title(f"Top {n} named entities")
    _save(fig, "entity_frequency.png")


def community_sentiment(top_n: int = 12):
    """Sentiment per discourse camp — communities rolled up by label
    (F1 fans, EV enthusiasts, Apple/design crowd, traders, ...). 100%-stacked
    so camps of different sizes are comparable; n = docs per camp."""
    df = load_csv("community_camps.csv")
    parts = [p for p in ("negative", "neutral", "positive") if p in df.columns]
    df = df.sort_values("n_docs", ascending=False).head(top_n).iloc[::-1]  # biggest on top
    labels = df["label"].tolist()
    totals = df[parts].sum(axis=1).replace(0, 1).values
    left = [0.0] * len(df)
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(df) + 1.8))
    for p in parts:
        frac = (df[p].values / totals)
        ax.barh(labels, frac, left=left, color=SENT_COLORS[p], label=p)
        left = [l + f for l, f in zip(left, frac)]
    for i, n in enumerate(df["n_docs"]):
        ax.text(1.01, i, f"n={int(n)}", va="center", fontsize=7, color="#444")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of documents")
    ax.set_title("Sentiment by discourse camp (community-level)")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
    _save(fig, "community_sentiment.png")


def community_network(top_n: int = 200):
    """Static interaction network of the most central accounts, coloured by
    community and sized by PageRank (matplotlib + networkx spring layout).

    The full graph is an 8k-node hairball, so we draw the subgraph induced by
    the top-N accounts by PageRank, which keeps the figure legible.
    """
    import networkx as nx
    G = nx.read_graphml(config.DATA_PROCESSED / "graph.graphml")
    if G.number_of_nodes() == 0:
        return
    comm = load_csv("nodes_communities.csv").set_index("node")["community_louvain"].to_dict()
    cent = load_csv("nodes_centrality.csv").set_index("node")["pagerank"].to_dict()

    top = sorted(G.nodes, key=lambda n: cent.get(n, 0) or 0, reverse=True)[:top_n]
    H = G.subgraph(top).to_undirected()
    pos = nx.spring_layout(H, seed=42, k=0.3)

    communities = [int(comm.get(n, 0) or 0) for n in H.nodes]
    sizes = [40 + 4000 * float(cent.get(n, 0) or 0) for n in H.nodes]

    fig, ax = plt.subplots(figsize=(11, 8.5))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.15, width=0.6, edge_color="#999")
    nodes = nx.draw_networkx_nodes(H, pos, ax=ax, node_color=communities, cmap="tab20",
                                   node_size=sizes, linewidths=0.3, edgecolors="white")
    # label only the dozen most central accounts so it stays readable
    top_labels = sorted(H.nodes, key=lambda n: cent.get(n, 0) or 0, reverse=True)[:12]
    nx.draw_networkx_labels(H, pos, labels={n: n for n in top_labels}, ax=ax, font_size=7)
    ax.set_title(f"Interaction network — top {H.number_of_nodes()} accounts by PageRank "
                 f"(colour = community)")
    ax.axis("off")
    _save(fig, "community_network.png")


def main() -> None:
    for fn in (sentiment_distribution, emotions, aspect_sentiment, timeline,
               wordclouds, centrality_top, entity_frequency,
               community_sentiment, community_network):
        _try(fn)


if __name__ == "__main__":
    main()
