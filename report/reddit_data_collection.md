# Reddit Data Collection — Methodology

*Web and Social Media Search and Analysis (BSc AI, UniMiB) — Track 1, LAB 2.*
*Topic: online reaction to the Ferrari **Luce** (Ferrari's first electric car).*

This document describes **exactly** how the Reddit dataset used in the project was
collected, so the procedure is reproducible and citable in the report. All behaviour
described here is implemented in [`src/collect_reddit.py`](../src/collect_reddit.py)
and parameterised in [`src/config.py`](../src/config.py).

---

## 1. Summary

| | |
|---|---|
| **Platform** | Reddit (public submissions + comments) |
| **Primary access method** | **Arctic-Shift** API — a free, no-account, actively-maintained Pushshift-style archive of Reddit |
| **Fallbacks** | Official Reddit API via **PRAW** (if credentials supplied) → **PullPush.io** (if Arctic-Shift is down) |
| **Subreddits** | r/Ferrari, r/cars, r/electricvehicles, r/formula1, r/stocks, r/wallstreetbets |
| **Search terms** | "Ferrari Luce", "Ferrari Elettrica", "Ferrari electric", "Ferrari EV", "Ferrari electric car" |
| **Time window** | since `2025-10-01` (Capital Markets Day) up to collection date |
| **Collected** | **374 submissions + 16,889 comments** (collected 2026-06-04) |
| **Outputs** | `data/processed/reddit_submissions.csv`, `data/processed/reddit_comments.csv` |

---

## 2. Why Arctic-Shift (and not Reddit directly)

The course (LAB 2) teaches collection through the official Reddit API (PRAW), which
requires registering a Reddit "script" app (client id + secret). To keep the pipeline
runnable **with no Reddit account at all** (e.g. on Colab, for graders), the collector
also supports two account-free archives. At collection time:

- **Reddit's own JSON endpoints** (`https://www.reddit.com/…​.json`) now return **HTTP 403**
  for unauthenticated / data-centre clients, so they are no longer a viable no-account path.
- **PullPush.io** (the commonly-cited Pushshift successor) was **unavailable** — its search
  endpoint timed out repeatedly (4 × 60 s) even though its front page responded.
- **Arctic-Shift** (`arctic-shift.photon-reddit.com`) responded normally and returns
  full Pushshift-schema records, so it was selected as the primary no-account source.

`main()` therefore auto-selects, in order of preference:

```
PRAW (official API, if REDDIT_CLIENT_ID/SECRET present)
  └─ else Arctic-Shift  (no account)
       └─ else PullPush.io  (no account, last resort)
```

> **Note on provenance.** Arctic-Shift and PullPush are *archives/mirrors* of Reddit, not
> Reddit's first-party API. They can lag slightly, may retain content later deleted on
> Reddit, and are not guaranteed 100 % complete versus the official API. Every collected
> row keeps its original `permalink`, so any item can be verified on reddit.com. For a
> first-party "ground-truth" pull, supply Reddit API credentials and the PRAW path is used.

---

## 3. What is queried

Defined in `src/config.py`:

- **Subreddits** (`SUBREDDITS`): chosen to span the relevant discourse communities —
  the brand (`r/Ferrari`), general car enthusiasts (`r/cars`), the EV angle
  (`r/electricvehicles`), motorsport fans (`r/formula1`), and the financial/market
  reaction (`r/stocks`, `r/wallstreetbets`).
- **Search terms** (`REDDIT_QUERIES`): `"Ferrari Luce"`, `"Ferrari Elettrica"`,
  `"Ferrari electric"`, `"Ferrari EV"`, `"Ferrari electric car"`.
- **Time window**: `SINCE_ISO = 2025-10-01T00:00:00Z` (the Capital Markets Day
  "Elettrica" preview, the widest meaningful window) through the collection date.
  `FOCUS_SINCE_ISO = 2026-05-20` marks the dense reveal window for later temporal analysis.

The collector iterates **every (subreddit × query) pair** and pulls matching submissions.

---

## 4. Collection procedure (Arctic-Shift path)

### 4.1 Endpoints

- Submissions: `GET https://arctic-shift.photon-reddit.com/api/posts/search`
- Comments: `GET https://arctic-shift.photon-reddit.com/api/comments/search`

A descriptive `User-Agent` header is sent on every request.

### 4.2 Submissions

For each subreddit `S` and query `Q`, the collector calls the posts endpoint with
`subreddit=S`, `query=Q`, `after=<epoch(SINCE_ISO)>`, `sort=desc`, `limit=100`
(the API maximum per page).

- **Pagination** is backward in time: after each page, the `before` cursor is set to
  `(oldest created_utc in the page) − 1`, and the next page is requested. Paging stops
  when a page returns fewer than 100 items or the per-query cap is reached
  (`REDDIT_MAX_PER_QUERY = 500`).
- **Politeness / robustness**: a `1.0 s` sleep separates requests, and every HTTP call is
  wrapped in exponential-backoff retry (`with_retries`, up to 5 attempts) to ride out
  transient errors and rate limits.

### 4.3 Relevance filtering

Archive full-text search is loose — in Italian, *luce* means "light", which produces
off-topic noise. Each candidate submission is kept **only if** its title+selftext is
relevant, per `_relevant()`:

> keep the post **iff** it mentions *"ferrari"* **and** either contains the exact pairing
> *"ferrari luce"* / *"luce ferrari"*, **or** contains an EV-context token from
> `RELEVANCE_ANY` = `{electric, elettrica, ev, battery, all-electric, jony ive, lovefrom,
> first electric, maranello ev}`.

For each query the collector logs `kept / fetched`, so the filter's effect is auditable
and the queries are tunable.

### 4.4 Comments

After submissions are collected and **de-duplicated by post id** (`dedup`), the collector
fetches the comment tree of each submission via the comments endpoint using `link_id=<post id>`
(the comment search API has no free-text `query`, so collection is per-submission).
Pagination is the same backward-`before` scheme. Caps: up to `400` submissions and `400`
comments per submission. Progress is logged every 25 submissions.

### 4.5 De-duplication

Submissions are de-duplicated by `id` after all (subreddit × query) pulls, so a post that
matches several queries is stored once; its `matched_query` field records the query that
first surfaced it.

---

## 5. Output schema

`data/processed/reddit_submissions.csv` — one row per submission:

`id`, `subreddit`, `matched_query`, `author`, `title`, `selftext`, `text`
(= `title + ". " + selftext`), `score`, `upvote_ratio`, `num_comments`, `created_at`
(UTC), `permalink`, `over_18`, `link_flair_text`, `source` (= `"reddit"`).

`data/processed/reddit_comments.csv` — one row per comment:

`id`, `submission_id`, `parent_id` (Reddit `t1_`/`t3_`-prefixed fullname),
`author`, `body`, `score`, `created_at` (UTC), `subreddit`, `source` (= `"reddit"`).

The `parent_id` field is what lets the network layer (`src/build_graph.py`) reconstruct
the **reply interaction graph** (comment author → parent author).

---

## 6. The collected dataset (snapshot 2026-06-04)

- **374** unique submissions, **16,889** comments.
- **294** unique submission authors, **8,320** unique comment authors.
- **Date span:** `2025-10-07` → `2026-06-04` (within the configured since-2025-10-01 window).

**By subreddit (submissions):**

| Subreddit | Submissions |
|---|---|
| r/Ferrari | 242 |
| r/cars | 62 |
| r/electricvehicles | 40 |
| r/formula1 | 15 |
| r/wallstreetbets | 11 |
| r/stocks | 4 |

**By surfacing query (submissions, post de-duplication):**

| Query | Submissions |
|---|---|
| Ferrari Luce | 274 |
| Ferrari electric | 53 |
| Ferrari EV | 38 |
| Ferrari Elettrica | 9 |

*(“Ferrari electric car” surfaced no additional unique posts after de-duplication.)*

---

## 7. Reproducing the collection

Locally (no account needed — uses Arctic-Shift):

```bash
python -m src.collect_reddit          # writes the two CSVs to data/processed/
```

To use the **official Reddit API** instead, set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`,
`REDDIT_USER_AGENT` (a read-only "script" app) — `main()` then uses PRAW automatically.

On Colab (`WSA_FerrariLuce_Colab.ipynb`): the §4.1 cell scrapes live by default
(`SCRAPE_FRESH = True`) and auto-falls back to a dataset bundled in the notebook if a
source is unavailable; set `SCRAPE_FRESH = False` to use the bundled data directly.

> Re-running scrapes **live** data, so counts will differ slightly from this snapshot as
> new posts appear and the archive updates.

---

## 8. Ethics & Terms of Service

- Only **public** Reddit content is collected; no private messages, no login-gated content.
- The official-API path is **read-only** (no posting, no voting), and account-free paths use
  public archives — no scraping of Reddit HTML and no circumvention of Reddit auth.
- Polite request spacing + backoff are used; proxies (if any) are for network routing only,
  **not** for evading rate limits or blocks.
- Usernames are personal data: they are used in aggregate for network/community analysis and
  are **not** singled out for individual profiling in the report. `.env` credentials are
  git-ignored and never committed.
