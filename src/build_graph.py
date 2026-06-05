"""LAB 3 — Social Network Analysis: build the directed interaction graph and
compute centrality measures.

WHY DIRECTED: "A replies to / mentions B" is asymmetric. Direction lets us
separate in-degree (attention received -> influence) from out-degree (activity),
which is what makes PageRank / directed betweenness meaningful (RQ1).
Community detection later symmetrizes to undirected (see communities.py).

Run:  python -m src.build_graph
Outputs: data/processed/graph.graphml , nodes_centrality.csv , graph_summary.txt
"""
from __future__ import annotations
import networkx as nx
import pandas as pd

from . import config
from .utils import log, load_csv, save_csv, is_bot

GRAPH_PATH = config.DATA_PROCESSED / "graph.graphml"


_BAD_NODES = {"[deleted]", "None", "nan", ""}


def _add_edge(G: nx.DiGraph, u, v, relation: str, platform: str) -> None:
    if pd.isna(u) or pd.isna(v):   # NaN from empty CSV fields / missing authors
        return
    u, v = str(u).strip(), str(v).strip()
    if u == v or u in _BAD_NODES or v in _BAD_NODES or is_bot(u) or is_bot(v):
        return
    if G.has_edge(u, v):
        G[u][v]["weight"] += 1
    else:
        G.add_edge(u, v, weight=1, relation=relation)
    for n in (u, v):
        G.nodes[n].setdefault("platform", platform)


def build_interaction_graph() -> nx.DiGraph:
    G = nx.DiGraph()

    # ---- Bluesky: mentions + replies ----
    try:
        bsky = load_csv("posts_bluesky.csv")
    except FileNotFoundError:
        bsky = pd.DataFrame()
    if not bsky.empty:
        uri2handle = dict(zip(bsky["uri"], bsky["author_handle"]))
        did2handle = dict(zip(bsky["author_did"], bsky["author_handle"]))
        for _, r in bsky.iterrows():
            src = r.get("author_handle")
            mentions = r.get("mention_dids")
            mentions = "" if pd.isna(mentions) else str(mentions)
            for did in (d.strip() for d in mentions.split(",")):
                if did:
                    _add_edge(G, src, did2handle.get(did, did), "mention", "bluesky")
            parent = r.get("reply_parent_uri")
            if isinstance(parent, str) and parent in uri2handle:
                _add_edge(G, src, uri2handle[parent], "reply", "bluesky")

    # ---- Reddit: comment author -> parent author ----
    try:
        subs = load_csv("reddit_submissions.csv")
        coms = load_csv("reddit_comments.csv")
    except FileNotFoundError:
        subs, coms = pd.DataFrame(), pd.DataFrame()
    if not coms.empty:
        sub_author = dict(zip(subs.get("id", []), subs.get("author", []))) if not subs.empty else {}
        com_author = dict(zip(coms["id"], coms["author"]))
        for _, c in coms.iterrows():
            src = c.get("author")
            pid = str(c.get("parent_id") or "")
            target = None
            if pid.startswith("t3_"):
                target = sub_author.get(pid[3:])
            elif pid.startswith("t1_"):
                target = com_author.get(pid[3:])
            _add_edge(G, src, target, "reply", "reddit")

    log.info("Interaction graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def compute_centralities(G: nx.DiGraph) -> pd.DataFrame:
    if G.number_of_nodes() == 0:
        return pd.DataFrame()
    indeg = nx.in_degree_centrality(G)
    outdeg = nx.out_degree_centrality(G)
    # betweenness can be costly; sample on large graphs
    k = min(G.number_of_nodes(), 400) if G.number_of_nodes() > 400 else None
    btw = nx.betweenness_centrality(G, k=k, weight="weight", seed=42)
    clo = nx.closeness_centrality(G)
    pr = nx.pagerank(G, weight="weight")
    try:
        eig = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig = {n: float("nan") for n in G.nodes}
    rows = [{
        "node": n,
        "platform": G.nodes[n].get("platform"),
        "in_degree": G.in_degree(n),
        "out_degree": G.out_degree(n),
        "in_degree_centrality": indeg.get(n),
        "out_degree_centrality": outdeg.get(n),
        "betweenness": btw.get(n),
        "closeness": clo.get(n),
        "pagerank": pr.get(n),
        "eigenvector": eig.get(n),
    } for n in G.nodes]
    df = pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)
    save_csv(df, "nodes_centrality.csv")
    return df


def graph_summary(G: nx.DiGraph) -> str:
    if G.number_of_nodes() == 0:
        return "empty graph"
    U = G.to_undirected()
    lines = [
        f"nodes: {G.number_of_nodes()}",
        f"edges: {G.number_of_edges()}",
        f"density: {nx.density(G):.5f}",
        f"transitivity (clustering): {nx.transitivity(U):.4f}",
        f"avg clustering: {nx.average_clustering(U):.4f}",
        f"reciprocity: {nx.reciprocity(G):.4f}",
        f"weakly connected components: {nx.number_weakly_connected_components(G)}",
        f"strongly connected components: {nx.number_strongly_connected_components(G)}",
        f"degree assortativity: {nx.degree_assortativity_coefficient(G):.4f}",
    ]
    txt = "\n".join(lines)
    (config.DATA_PROCESSED / "graph_summary.txt").write_text(txt)
    log.info("Graph summary:\n%s", txt)
    return txt


def save_graph(G: nx.DiGraph) -> None:
    nx.write_graphml(G, GRAPH_PATH)
    log.info("graph -> %s (open in Gephi)", GRAPH_PATH)


def load_graph() -> nx.DiGraph:
    return nx.read_graphml(GRAPH_PATH)


def main() -> None:
    G = build_interaction_graph()
    compute_centralities(G)
    graph_summary(G)
    save_graph(G)


if __name__ == "__main__":
    main()
