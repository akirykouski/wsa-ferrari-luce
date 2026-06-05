"""LAB 4 — Community detection on the (symmetrized) interaction graph.

Louvain and greedy-modularity partitions are compared via modularity Q;
assortativity tests the echo-chamber hypothesis (RQ2). Community labels are
attached back to nodes so the content layer can compute per-community sentiment.

Run:  python -m src.communities  (requires src.build_graph to have run first)
Outputs: data/processed/nodes_communities.csv , communities_summary.txt
"""
from __future__ import annotations
from collections import Counter

import networkx as nx
import pandas as pd

from . import config
from .utils import log, save_csv
from .build_graph import load_graph


def to_undirected_weighted(G: nx.DiGraph) -> nx.Graph:
    """Collapse reciprocal directed edges, summing weights (for modularity)."""
    U = nx.Graph()
    U.add_nodes_from(G.nodes(data=True))
    for u, v, d in G.edges(data=True):
        w = d.get("weight", 1)
        if U.has_edge(u, v):
            U[u][v]["weight"] += w
        else:
            U.add_edge(u, v, weight=w)
    return U


def detect(U: nx.Graph) -> tuple[dict, dict, dict]:
    """Return node->community dicts for louvain & greedy, plus a metrics dict."""
    louvain = nx.community.louvain_communities(U, weight="weight", seed=42)
    greedy = nx.community.greedy_modularity_communities(U, weight="weight")

    def as_map(communities):
        return {n: i for i, com in enumerate(communities) for n in com}

    lmap, gmap = as_map(louvain), as_map(greedy)
    metrics = {
        "louvain_n_communities": len(louvain),
        "louvain_modularity": nx.community.modularity(U, louvain, weight="weight"),
        "greedy_n_communities": len(greedy),
        "greedy_modularity": nx.community.modularity(U, greedy, weight="weight"),
        "degree_assortativity": nx.degree_assortativity_coefficient(U),
    }
    return lmap, gmap, metrics


def merge_small(comm_map: dict, min_size: int = 5) -> dict:
    """Relabel members of any community below `min_size` to a single 'other'
    bucket (community -1), so tiny (often singleton) communities don't clutter
    the discourse-camp summary or the network figure."""
    sizes = Counter(comm_map.values())
    return {n: (c if sizes[c] >= min_size else -1) for n, c in comm_map.items()}


def label_top_members(U: nx.Graph, comm_map: dict, top: int = 8) -> dict[int, list[str]]:
    """Most central member handles per community (for naming the discourse camps)."""
    deg = dict(U.degree(weight="weight"))
    by_comm: dict[int, list[str]] = {}
    for node, c in comm_map.items():
        by_comm.setdefault(c, []).append(node)
    return {
        c: sorted(members, key=lambda n: deg.get(n, 0), reverse=True)[:top]
        for c, members in by_comm.items()
    }


def main() -> None:
    G = load_graph()
    U = to_undirected_weighted(G)
    if U.number_of_nodes() == 0:
        log.warning("Empty graph; run collectors + build_graph first.")
        return
    lmap, gmap, metrics = detect(U)
    # Merge sub-threshold communities into an "other" bucket (-1) before they
    # reach the node table, the camp summary and the network figure.
    lmap = merge_small(lmap, config.MIN_COMMUNITY_SIZE)
    gmap = merge_small(gmap, config.MIN_COMMUNITY_SIZE)
    metrics["min_community_size"] = config.MIN_COMMUNITY_SIZE
    metrics["louvain_communities_ge_min"] = len({c for c in lmap.values() if c != -1})

    df = pd.DataFrame({
        "node": list(U.nodes),
        "platform": [U.nodes[n].get("platform") for n in U.nodes],
        "community_louvain": [lmap.get(n) for n in U.nodes],
        "community_greedy": [gmap.get(n) for n in U.nodes],
        "weighted_degree": [U.degree(n, weight="weight") for n in U.nodes],
    }).sort_values(["community_louvain", "weighted_degree"], ascending=[True, False])
    save_csv(df, "nodes_communities.csv")

    tops = label_top_members(U, lmap)
    lines = [f"{k}: {v}" for k, v in metrics.items()]
    lines.append(f"\nTop members per Louvain community (n>={config.MIN_COMMUNITY_SIZE}; "
                 "name the camps from these):")
    n_other = sum(1 for x in lmap.values() if x == -1)
    for c, members in sorted(tops.items()):
        if c == -1:
            continue  # the merged "other" bucket isn't a real camp
        lines.append(f"  community {c} (n={sum(1 for x in lmap.values() if x == c)}): {members}")
    if n_other:
        lines.append(f"  other (merged communities with n<{config.MIN_COMMUNITY_SIZE}): "
                     f"{n_other} accounts")
    txt = "\n".join(lines)
    (config.DATA_PROCESSED / "communities_summary.txt").write_text(txt)
    log.info("Communities:\n%s", txt)


if __name__ == "__main__":
    main()
