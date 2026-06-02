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
    fig, ax = plt.subplots(figsize=(6, 4))
    means.plot(kind="bar", ax=ax, color="#9467bd")
    ax.set_title("Mean NRC emotion intensity")
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
    tl["created_at"] = pd.to_datetime(tl["created_at"], errors="coerce")
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.bar(tl["created_at"], tl["n_docs"], color="#aec7e8", label="volume")
    ax1.set_ylabel("posts / day", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(tl["created_at"], tl["mean_compound"], color="#d62728", marker="o", label="mean sentiment")
    ax2.axhline(0, color="grey", lw=0.6, ls="--")
    ax2.set_ylabel("mean VADER compound", color="#d62728")
    for date, lbl in config.EVENT_DATES:
        try:
            ax1.axvline(pd.Timestamp(date, tz="UTC"), color="black", lw=0.8, ls=":")
        except Exception:
            pass
    ax1.set_title("Volume & sentiment over time")
    _save(fig, "sentiment_timeline.png")


def wordclouds():
    try:
        from wordcloud import WordCloud
    except Exception:
        return
    src = "documents_enriched.csv" if (config.DATA_PROCESSED / "documents_enriched.csv").exists() else "documents_sentiment.csv"
    df = load_csv(src)
    col = next((c for c in ("sentiment_corrected", "transformer_label", "vader_label") if c in df.columns), None)
    for sentiment in ("negative", "positive"):
        text = " ".join(df.loc[df[col] == sentiment, "text"].astype(str))
        if len(text) < 50:
            continue
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
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


def community_network():
    """Interactive hairball coloured by community (pyvis -> html)."""
    try:
        import networkx as nx
        from pyvis.network import Network
    except Exception:
        return
    G = nx.read_graphml(config.DATA_PROCESSED / "graph.graphml")
    comm = load_csv("nodes_communities.csv").set_index("node")["community_louvain"].to_dict()
    cent = load_csv("nodes_centrality.csv").set_index("node")["pagerank"].to_dict()
    # cdn_resources="remote" -> pyvis links JS/CSS from a CDN instead of dumping
    # a local lib/ folder next to the html.
    net = Network(height="750px", width="100%", bgcolor="#111", font_color="white",
                  cdn_resources="remote")
    palette = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
               "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]
    for node in G.nodes:
        c = comm.get(node, 0) or 0
        net.add_node(node, label=str(node),
                     color=palette[int(c) % len(palette)],
                     size=8 + 600 * float(cent.get(node, 0) or 0))
    for u, v, d in G.edges(data=True):
        net.add_edge(u, v, value=d.get("weight", 1))
    out = config.FIGURES / "community_network.html"
    net.write_html(str(out))
    log.info("interactive network -> %s", out)


def main() -> None:
    for fn in (sentiment_distribution, emotions, aspect_sentiment, timeline,
               wordclouds, centrality_top, entity_frequency, community_network):
        _try(fn)


if __name__ == "__main__":
    main()
