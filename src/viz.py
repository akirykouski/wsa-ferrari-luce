from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="notebook")
except Exception:
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
    except Exception as e:
        log.warning("%s failed: %s", fn.__name__, e)


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
    if means.sum() == 0:
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
    tl["sum_compound"] = tl["mean_compound"].fillna(0) * tl["n_docs"]
    wk = tl.resample("W").agg(n_docs=("n_docs", "sum"),
                              sum_compound=("sum_compound", "sum"))
    wk["mean_compound"] = (wk["sum_compound"] / wk["n_docs"]).where(wk["n_docs"] > 0)
    active = wk[wk["n_docs"] > 0]

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
    except Exception:
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

    for sentiment in sentiments:
        text = " ".join(df.loc[df[col] == sentiment, "text"])
        if len(text) < 50:
            continue
        wc = WordCloud(width=800, height=400, background_color="white",
                       stopwords=stop, collocations=False).generate(text)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc); ax.axis("off"); ax.set_title(f"{sentiment} posts")
        _save(fig, f"wordcloud_{sentiment}.png")


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
    df = load_csv("community_camps.csv")
    parts = [p for p in ("negative", "neutral", "positive") if p in df.columns]
    df = df.sort_values("n_docs", ascending=False).head(top_n).iloc[::-1]
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


def community_structure(top_communities: int = 10, max_per_community: int = 150):
    """Community map where every user is drawn at the SAME size and coloured by
    community, so the structure is clear regardless of any user's PageRank — a quiet
    member is as visible as a hub. The largest communities are each laid out as their
    own cluster (arranged on a ring); members are sampled to keep the figure legible.
    """
    import math
    import networkx as nx
    G = nx.read_graphml(config.DATA_PROCESSED / "graph.graphml")
    if G.number_of_nodes() == 0:
        return
    comm = load_csv("nodes_communities.csv").set_index("node")["community_louvain"].to_dict()
    try:  # human labels per community, if community_sentiment has run
        labels = load_csv("community_sentiment.csv").set_index("community")["label"].to_dict()
    except Exception:  # noqa: BLE001
        labels = {}

    from collections import Counter
    sizes = Counter(comm.get(n) for n in G.nodes if comm.get(n) is not None)
    kept = [c for c, _ in sizes.most_common(top_communities)]

    U = G.to_undirected()
    pos, node_comm, drawn = {}, {}, []
    R = 10.0
    for i, c in enumerate(kept):
        members = [n for n in U.nodes if comm.get(n) == c]
        if len(members) > max_per_community:
            members = sorted(members)[:max_per_community]   # deterministic, PageRank-blind
        sub = U.subgraph(members)
        local = nx.spring_layout(sub, seed=42, k=0.6)
        ang = 2 * math.pi * i / len(kept)
        cx, cy = R * math.cos(ang), R * math.sin(ang)
        for n, (x, y) in local.items():
            pos[n] = (cx + 3.0 * x, cy + 3.0 * y)
            node_comm[n] = i
            drawn.append(n)
    H = U.subgraph(drawn)

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.08, width=0.4, edge_color="#999")
    cmap = plt.cm.tab10 if len(kept) <= 10 else plt.cm.tab20
    nx.draw_networkx_nodes(H, pos, ax=ax, nodelist=drawn,
                           node_color=[node_comm[n] for n in drawn], cmap=cmap,
                           vmin=0, vmax=max(9, len(kept) - 1),
                           node_size=28, linewidths=0.2, edgecolors="white")
    for i, c in enumerate(kept):
        ang = 2 * math.pi * i / len(kept)
        lab = labels.get(c, f"community {c}")
        ax.text(1.18 * R * math.cos(ang), 1.18 * R * math.sin(ang),
                f"{lab}\n(n={sizes[c]})", ha="center", va="center",
                fontsize=8, fontweight="bold")
    ax.set_title(f"Community structure — every user equal size, coloured by community "
                 f"(top {len(kept)} communities; PageRank-independent)")
    ax.margins(0.12)
    ax.axis("off")
    _save(fig, "community_structure.png")


def community_network(top_n: int = 200):
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
    top_labels = sorted(H.nodes, key=lambda n: cent.get(n, 0) or 0, reverse=True)[:12]
    nx.draw_networkx_labels(H, pos, labels={n: n for n in top_labels}, ax=ax, font_size=7)
    ax.set_title(f"Interaction network — top {H.number_of_nodes()} accounts by PageRank "
                 f"(colour = community)")
    ax.axis("off")
    _save(fig, "community_network.png")


def main() -> None:
    for fn in (sentiment_distribution, emotions, aspect_sentiment, timeline,
               wordclouds, centrality_top, entity_frequency,
               community_sentiment, community_structure, community_network):
        _try(fn)


if __name__ == "__main__":
    main()
