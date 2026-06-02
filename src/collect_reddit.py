"""LAB 2 — Reddit collector.

Primary path: PRAW in read-only OAuth (client id + secret + user agent).
Fallback:     PullPush.io (free, no key) for time-sliced historical pulls.

Key gotchas handled here:
  * comment trees are lazy  -> submission.comments.replace_more(limit=0)
  * listing 1000-item cap   -> diversify across subreddits x sort x time_filter

Run:  python -m src.collect_reddit
Outputs:  data/processed/reddit_submissions.csv , reddit_comments.csv
"""
from __future__ import annotations
from datetime import datetime, timezone
import requests
import pandas as pd

from . import config
from .utils import log, require_env, with_retries, save_csv, dedup


# ----------------------------------------------------------------------------
# PRAW (primary)
# ----------------------------------------------------------------------------
def get_reddit():
    creds = require_env("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT")
    import praw
    reddit = praw.Reddit(
        client_id=creds["REDDIT_CLIENT_ID"],
        client_secret=creds["REDDIT_CLIENT_SECRET"],
        user_agent=creds["REDDIT_USER_AGENT"],
        check_for_async=False,
    )
    reddit.read_only = True
    log.info("Reddit client ready (read_only=%s)", reddit.read_only)
    return reddit


def _ts(utc) -> str | None:
    try:
        return datetime.fromtimestamp(float(utc), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _flatten_submission(s, subreddit: str, query: str) -> dict:
    title = getattr(s, "title", "") or ""
    body = getattr(s, "selftext", "") or ""
    author = getattr(s, "author", None)
    return {
        "id": getattr(s, "id", None),
        "subreddit": subreddit,
        "matched_query": query,
        "author": str(author) if author else "[deleted]",
        "title": title,
        "selftext": body,
        "text": (title + ". " + body).strip(),
        "score": getattr(s, "score", 0),
        "upvote_ratio": getattr(s, "upvote_ratio", None),
        "num_comments": getattr(s, "num_comments", 0),
        "created_at": _ts(getattr(s, "created_utc", None)),
        "permalink": getattr(s, "permalink", None),
        "over_18": getattr(s, "over_18", None),
        "link_flair_text": getattr(s, "link_flair_text", None),
        "source": "reddit",
    }


def collect_submissions(reddit, queries=None, subreddits=None) -> pd.DataFrame:
    queries = queries or config.REDDIT_QUERIES
    subreddits = subreddits or config.SUBREDDITS
    rows: list[dict] = []
    for sub in subreddits:
        sr = reddit.subreddit(sub)
        for q in queries:
            try:
                gen = sr.search(q, sort="relevance",
                                time_filter=config.REDDIT_TIME_FILTER,
                                limit=config.REDDIT_MAX_PER_QUERY)
                n0 = len(rows)
                for s in gen:
                    rows.append(_flatten_submission(s, sub, q))
                log.info("  r/%-16s %-22s -> %d", sub, q, len(rows) - n0)
            except Exception as e:  # noqa: BLE001
                log.warning("  search failed r/%s '%s': %s", sub, q, e)
    rows = dedup(rows, key="id")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    save_csv(df, "reddit_submissions.csv")
    return df


def collect_comments(reddit, submission_ids: list[str]) -> pd.DataFrame:
    """Flatten comment trees. replace_more(limit=0) so no replies are dropped."""
    rows: list[dict] = []
    for i, sid in enumerate(submission_ids, 1):
        try:
            sub = reddit.submission(id=sid)
            with_retries(sub.comments.replace_more, limit=0)
            for c in sub.comments.list():
                author = getattr(c, "author", None)
                rows.append({
                    "id": getattr(c, "id", None),
                    "submission_id": sid,
                    "parent_id": getattr(c, "parent_id", None),  # t1_/t3_ prefixed
                    "author": str(author) if author else "[deleted]",
                    "body": getattr(c, "body", "") or "",
                    "score": getattr(c, "score", 0),
                    "created_at": _ts(getattr(c, "created_utc", None)),
                    "subreddit": str(getattr(c, "subreddit", "")),
                    "source": "reddit",
                })
        except Exception as e:  # noqa: BLE001
            log.warning("  comments failed for %s: %s", sid, e)
        if i % 25 == 0:
            log.info("  comments: %d/%d submissions processed", i, len(submission_ids))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    save_csv(df, "reddit_comments.csv")
    return df


# ----------------------------------------------------------------------------
# PullPush.io (fallback) — free, no key. Limits: 15/30 rpm, 1000/hour.
# ----------------------------------------------------------------------------
PULLPUSH = "https://api.pullpush.io/reddit/search"


def pullpush_search(kind: str, query: str, subreddit: str | None = None,
                    since: int | None = None, until: int | None = None,
                    size: int = config.PULLPUSH_PAGE_SIZE) -> list[dict]:
    """kind in {'submission','comment'}. since/until are epoch seconds."""
    params = {"q": query, "size": size, "sort": "desc"}
    if subreddit:
        params["subreddit"] = subreddit
    if since:
        params["since"] = since
    if until:
        params["until"] = until
    resp = with_retries(requests.get, f"{PULLPUSH}/{kind}/", params=params, timeout=60)
    resp.raise_for_status()
    return resp.json().get("data", [])


def main() -> None:
    reddit = get_reddit()
    subs = collect_submissions(reddit)
    log.info("Collected %d unique submissions.", len(subs))
    if not subs.empty:
        comments = collect_comments(reddit, subs["id"].dropna().tolist())
        log.info("Collected %d comments.", len(comments))


if __name__ == "__main__":
    main()
