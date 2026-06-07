from __future__ import annotations
import os
import time
from datetime import datetime, timezone
import requests
import pandas as pd

from . import config
from .utils import log, require_env, with_retries, save_csv, dedup, load_env


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
            except Exception as e:  
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
                    "parent_id": getattr(c, "parent_id", None),  
                    "author": str(author) if author else "[deleted]",
                    "body": getattr(c, "body", "") or "",
                    "score": getattr(c, "score", 0),
                    "created_at": _ts(getattr(c, "created_utc", None)),
                    "subreddit": str(getattr(c, "subreddit", "")),
                    "source": "reddit",
                })
        except Exception as e:  
            log.warning("  comments failed for %s: %s", sid, e)
        if i % 25 == 0:
            log.info("  comments: %d/%d submissions processed", i, len(submission_ids))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    save_csv(df, "reddit_comments.csv")
    return df



PULLPUSH = "https://api.pullpush.io/reddit/search"
_PP_SLEEP = 2.5  # seconds between requests to stay under the rate limit


def _iso_to_epoch(iso: str | None) -> int | None:
    if not iso:
        return None
    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


def pullpush_search(kind: str, **params) -> list[dict]:
    """kind in {'submission','comment'}. Pass q / subreddit / link_id / after / before."""
    params.setdefault("size", config.PULLPUSH_PAGE_SIZE)
    params.setdefault("sort", "desc")
    clean = {k: v for k, v in params.items() if v is not None}
    resp = with_retries(requests.get, f"{PULLPUSH}/{kind}/", params=clean, timeout=60,
                        exceptions=(requests.RequestException,))
    resp.raise_for_status()
    return resp.json().get("data", [])


def _relevant(text: str) -> bool:
    """Filter loose full-text noise: require the exact 'ferrari luce' pairing,
    or Ferrari mentioned together with an EV-context token (config.RELEVANCE_ANY)."""
    t = (text or "").lower()
    if "ferrari" not in t:
        return False
    if "ferrari luce" in t or "luce ferrari" in t:
        return True
    return any(tok in t for tok in config.RELEVANCE_ANY)


def _flatten_pp_submission(d: dict, query: str) -> dict:
    title = d.get("title", "") or ""
    body = d.get("selftext", "") or ""
    return {
        "id": d.get("id"),
        "subreddit": d.get("subreddit"),
        "matched_query": query,
        "author": d.get("author") or "[deleted]",
        "title": title,
        "selftext": body,
        "text": (title + ". " + body).strip(),
        "score": d.get("score", 0),
        "upvote_ratio": d.get("upvote_ratio"),
        "num_comments": d.get("num_comments", 0),
        "created_at": _ts(d.get("created_utc")),
        "permalink": d.get("permalink"),
        "over_18": d.get("over_18"),
        "link_flair_text": d.get("link_flair_text"),
        "source": "reddit",
    }


def _paged(kind: str, base_params: dict, max_items: int) -> list[dict]:
    """Backward-paginate PullPush by moving `before` to the oldest item each page."""
    out, before = [], None
    while len(out) < max_items:
        data = pullpush_search(kind, before=before, **base_params)
        if not data:
            break
        out.extend(data)
        try:
            before = min(int(x["created_utc"]) for x in data if x.get("created_utc")) - 1
        except (ValueError, KeyError):
            break
        time.sleep(_PP_SLEEP)
        if len(data) < config.PULLPUSH_PAGE_SIZE:
            break
    return out[:max_items]


def pullpush_collect_submissions(queries=None, max_per_query=config.REDDIT_MAX_PER_QUERY,
                                 since_iso=config.SINCE_ISO) -> pd.DataFrame:
    queries = queries or config.REDDIT_QUERIES
    after = _iso_to_epoch(since_iso)
    rows: list[dict] = []
    for q in queries:
        try:
            data = _paged("submission", {"q": q, "after": after}, max_per_query)
            kept = [_flatten_pp_submission(d, q) for d in data
                    if _relevant(f"{d.get('title','')} {d.get('selftext','')}")]
            rows.extend(kept)
            log.info("  PullPush submission '%s' -> %d kept / %d fetched", q, len(kept), len(data))
        except Exception as e:  
            log.warning("  PullPush submission '%s' failed: %s", q, e)
    rows = dedup(rows, key="id")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    save_csv(df, "reddit_submissions.csv")
    return df


def pullpush_collect_comments(submission_ids, max_submissions=150,
                              max_per_submission=300) -> pd.DataFrame:
    rows: list[dict] = []
    ids = list(submission_ids)[:max_submissions]
    if len(submission_ids) > max_submissions:
        log.warning("comments capped to %d/%d submissions (rate limit).",
                    max_submissions, len(submission_ids))
    for i, sid in enumerate(ids, 1):
        try:
            data = _paged("comment", {"link_id": sid}, max_per_submission)
            for c in data:
                rows.append({
                    "id": c.get("id"),
                    "submission_id": sid,
                    "parent_id": c.get("parent_id"),
                    "author": c.get("author") or "[deleted]",
                    "body": c.get("body", "") or "",
                    "score": c.get("score", 0),
                    "created_at": _ts(c.get("created_utc")),
                    "subreddit": c.get("subreddit"),
                    "source": "reddit",
                })
        except Exception as e:  
            log.warning("  PullPush comments for %s failed: %s", sid, e)
        if i % 25 == 0:
            log.info("  comments: %d/%d submissions", i, len(ids))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    save_csv(df, "reddit_comments.csv")
    return df



ARCTIC = "https://arctic-shift.photon-reddit.com/api"
_AS_HEADERS = {"User-Agent": "wsa-ferrari-luce/0.1 (research; no-account collector)"}
_AS_SLEEP = 1.0          
_AS_PAGE = 100           


def arctic_search(kind: str, **params) -> list[dict]:
    """kind in {'posts','comments'}. Returns raw Pushshift-style records."""
    params.setdefault("limit", _AS_PAGE)
    clean = {k: v for k, v in params.items() if v is not None}
    resp = with_retries(requests.get, f"{ARCTIC}/{kind}/search", params=clean,
                        headers=_AS_HEADERS, timeout=60,
                        exceptions=(requests.RequestException,))
    resp.raise_for_status()
    return resp.json().get("data") or []


def _arctic_paged(kind: str, base_params: dict, max_items: int) -> list[dict]:
    """Backward-paginate by moving `before` just past the oldest item each page."""
    out: list[dict] = []
    before = base_params.get("before")
    while len(out) < max_items:
        page = arctic_search(kind, sort="desc",
                             **{**base_params, "before": before, "limit": _AS_PAGE})
        if not page:
            break
        out.extend(page)
        try:
            before = min(int(x["created_utc"]) for x in page if x.get("created_utc")) - 1
        except (ValueError, KeyError):
            break
        time.sleep(_AS_SLEEP)
        if len(page) < _AS_PAGE:
            break
    return out[:max_items]


def arctic_collect_submissions(queries=None, subreddits=None,
                               max_per_query=config.REDDIT_MAX_PER_QUERY,
                               since_iso=config.SINCE_ISO) -> pd.DataFrame:
    queries = queries or config.REDDIT_QUERIES
    subreddits = subreddits or config.SUBREDDITS
    after = _iso_to_epoch(since_iso)
    rows: list[dict] = []
    for sub in subreddits:
        for q in queries:
            try:
                data = _arctic_paged("posts", {"subreddit": sub, "query": q,
                                               "after": after}, max_per_query)
                kept = [_flatten_pp_submission(d, q) for d in data
                        if _relevant(f"{d.get('title','')} {d.get('selftext','')}")]
                rows.extend(kept)
                log.info("  Arctic r/%-16s %-22s -> %d kept / %d fetched",
                         sub, q, len(kept), len(data))
            except Exception as e:  
                log.warning("  Arctic r/%s '%s' failed: %s", sub, q, e)
    rows = dedup(rows, key="id")
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    save_csv(df, "reddit_submissions.csv")
    return df


def arctic_collect_comments(submission_ids, max_submissions=400,
                            max_per_submission=400) -> pd.DataFrame:
    rows: list[dict] = []
    ids = list(submission_ids)[:max_submissions]
    if len(submission_ids) > max_submissions:
        log.warning("comments capped to %d/%d submissions.", max_submissions, len(submission_ids))
    for i, sid in enumerate(ids, 1):
        try:
            data = _arctic_paged("comments", {"link_id": sid}, max_per_submission)
            for c in data:
                rows.append({
                    "id": c.get("id"),
                    "submission_id": sid,
                    "parent_id": c.get("parent_id"),
                    "author": c.get("author") or "[deleted]",
                    "body": c.get("body", "") or "",
                    "score": c.get("score", 0),
                    "created_at": _ts(c.get("created_utc")),
                    "subreddit": c.get("subreddit"),
                    "source": "reddit",
                })
        except Exception as e: 
            log.warning("  Arctic comments for %s failed: %s", sid, e)
        if i % 25 == 0:
            log.info("  comments: %d/%d submissions", i, len(ids))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    save_csv(df, "reddit_comments.csv")
    return df



def main() -> None:
    load_env()
    have_api = all(os.environ.get(k) for k in
                   ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET", "REDDIT_USER_AGENT"))
    if have_api:
        log.info("Reddit API creds found -> using PRAW.")
        reddit = get_reddit()
        subs = collect_submissions(reddit)
        if not subs.empty:
            collect_comments(reddit, subs["id"].dropna().tolist())
        return

    # No account: Arctic-Shift first, PullPush as fallback.
    log.info("No Reddit API creds -> using Arctic-Shift (no account needed).")
    collect_comments_for = arctic_collect_comments
    try:
        subs = arctic_collect_submissions()
    except Exception as e:  
        log.warning("Arctic-Shift failed (%s); trying PullPush.io.", e)
        subs = pd.DataFrame()
    if subs.empty:
        log.warning("Arctic-Shift returned no submissions; falling back to PullPush.io.")
        subs = pullpush_collect_submissions()
        collect_comments_for = pullpush_collect_comments
    log.info("Collected %d unique submissions.", len(subs))
    if not subs.empty:
        comments = collect_comments_for(subs["id"].dropna().tolist())
        log.info("Collected %d comments.", len(comments))


if __name__ == "__main__":
    main()
