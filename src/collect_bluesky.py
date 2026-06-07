"""
Collects posts matching the topic queries (with engagement + facets + thread
links) and optionally the follow graph of key accounts.

"""
from __future__ import annotations
import os
import time
import pandas as pd

from . import config
from .utils import log, require_env, with_retries, save_csv, dedup, load_env


def have_credentials() -> bool:
    """True if both Bluesky env vars are set (so collection can run).

    Checks env only — does NOT import `atproto` — so a no-credentials Colab
    run can skip Bluesky cleanly even if the client library isn't installed.
    """
    load_env()
    return all(os.environ.get(k) for k in ("BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD"))


def get_client():
    """Authenticate with a Bluesky App Password (never the main password)."""
    creds = require_env("BLUESKY_HANDLE", "BLUESKY_APP_PASSWORD")
    from atproto import Client
    client = Client()
    with_retries(client.login, creds["BLUESKY_HANDLE"], creds["BLUESKY_APP_PASSWORD"])
    log.info("Bluesky login OK as %s", creds["BLUESKY_HANDLE"])
    return client


def _facets(record) -> tuple[list[str], list[str], list[str]]:
    """Extract (hashtags, mention DIDs, links) from a post record's facets."""
    tags, mentions, links = [], [], []
    for facet in (getattr(record, "facets", None) or []):
        for feat in (getattr(facet, "features", None) or []):
            tag = getattr(feat, "tag", None)
            did = getattr(feat, "did", None)
            uri = getattr(feat, "uri", None)
            if tag:
                tags.append(tag.lower())
            if did:
                mentions.append(did)
            if uri:
                links.append(uri)
    return tags, mentions, links


def _flatten(post) -> dict:
    record = getattr(post, "record", None)
    author = getattr(post, "author", None)
    tags, mentions, links = _facets(record)
    reply = getattr(record, "reply", None)
    return {
        "uri": getattr(post, "uri", None),
        "cid": getattr(post, "cid", None),
        "created_at": getattr(record, "created_at", None),
        "text": getattr(record, "text", "") or "",
        "author_handle": getattr(author, "handle", None),
        "author_did": getattr(author, "did", None),
        "author_display_name": getattr(author, "display_name", None),
        "like_count": getattr(post, "like_count", 0) or 0,
        "repost_count": getattr(post, "repost_count", 0) or 0,
        "reply_count": getattr(post, "reply_count", 0) or 0,
        "quote_count": getattr(post, "quote_count", 0) or 0,
        "langs": ",".join(getattr(record, "langs", None) or []),
        "hashtags": ",".join(tags),
        "mention_dids": ",".join(mentions),
        "links": ",".join(links),
        "reply_parent_uri": getattr(getattr(reply, "parent", None), "uri", None) if reply else None,
        "reply_root_uri": getattr(getattr(reply, "root", None), "uri", None) if reply else None,
    }


def search_query(client, query: str, max_results: int = config.BLUESKY_MAX_PER_QUERY,
                 since: str | None = config.SINCE_ISO, until: str | None = config.UNTIL_ISO) -> list[dict]:
    """Cursor-paginated search_posts for a single query (LAB 2)."""
    out: list[dict] = []
    cursor = None
    while len(out) < max_results:
        params = {"q": query, "limit": config.BLUESKY_PAGE_SIZE, "sort": "latest"}
        if cursor:
            params["cursor"] = cursor
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        resp = with_retries(client.app.bsky.feed.search_posts, params)
        posts = getattr(resp, "posts", None) or []
        if not posts:
            break
        out.extend(_flatten(p) for p in posts)
        cursor = getattr(resp, "cursor", None)
        if not cursor:
            break
        time.sleep(0.4)  # be polite
    log.info("  query %-28s -> %d posts", query, len(out))
    return out[:max_results]


def collect_posts(queries: list[str] | None = None) -> pd.DataFrame:
    client = get_client()
    queries = queries or config.BLUESKY_QUERIES
    rows: list[dict] = []
    for q in queries:
        rows.extend(search_query(client, q))
    rows = dedup(rows, key="uri")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["source"] = "bluesky"
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    save_csv(df, "posts_bluesky.csv")
    return df


def collect_follow_edges(accounts: list[str] | None = None) -> pd.DataFrame:
    """Optional follower/follows ego-networks for key accounts."""
    accounts = accounts or config.KEY_BLUESKY_ACCOUNTS
    if not accounts:
        log.info("No KEY_BLUESKY_ACCOUNTS configured; skipping follow graph.")
        return pd.DataFrame(columns=["source", "target", "relation"])
    client = get_client()
    edges: list[dict] = []
    for actor in accounts:
        for relation, endpoint, field in (
            ("follows", client.app.bsky.graph.get_follows, "follows"),
            ("followed_by", client.app.bsky.graph.get_followers, "followers"),
        ):
            cursor = None
            while True:
                params = {"actor": actor, "limit": 100}
                if cursor:
                    params["cursor"] = cursor
                resp = with_retries(endpoint, params)
                people = getattr(resp, field, None) or []
                for p in people:
                    other = getattr(p, "handle", None)
                    if relation == "follows":
                        edges.append({"source": actor, "target": other, "relation": "follows"})
                    else:
                        edges.append({"source": other, "target": actor, "relation": "follows"})
                cursor = getattr(resp, "cursor", None)
                if not cursor:
                    break
    df = pd.DataFrame(edges)
    save_csv(df, "edges_follows.csv")
    return df


def main() -> None:
    if not have_credentials():
        log.warning(
            "No Bluesky credentials (BLUESKY_HANDLE / BLUESKY_APP_PASSWORD) -> "
            "skipping Bluesky. Reddit/PullPush still runs, so the pipeline "
            "completes on Reddit data alone. Set both to include Bluesky."
        )
        return
    df = collect_posts()
    log.info("Collected %d unique Bluesky posts.", len(df))
    collect_follow_edges()


if __name__ == "__main__":
    main()
