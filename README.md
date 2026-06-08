# Social Media Analysis of the Ferrari Luce

**Network Analysis + Content Analysis + Visualization** of the online reaction to Ferrari's first electric car, the **Luce** (revealed 25 May 2026).


---

## 1. Setup

> The ML stack (torch/transformers/spaCy/BERTopic) ships wheels for **Python
> 3.10–3.12**, matching Google Colab. Use a 3.11 venv; newer interpreters may
> lack wheels.

```bash
cd WSA_FerrariLuce
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # NLTK data downloads on first use
cp .env.example .env                       # optional — collectors run with no creds
```

The pipeline lives entirely in `src/`; run each
stage as a module, or open `WSA_FerrariLuce_Colab.ipynb`, which imports `src/` directly.

### Credentials (`.env`)
- **Bluesky:** create an *App Password* (bsky.app → Settings → App Passwords).
- **Reddit:** register a **script** app at <https://www.reddit.com/prefs/apps>.
  We use **read-only** mode — only `client_id`, `client_secret`, `user_agent`
  are required (no Reddit password).
  - **No Reddit account / app creation blocked?** Leave the `REDDIT_*` vars unset
    and the collector falls back to a free, **no-account / no-key** archive:
    **Arctic-Shift** (the actively-maintained Pushshift mirror) first, then
    **PullPush.io** if Arctic-Shift is unavailable. `main()` auto-selects
    PRAW → Arctic-Shift → PullPush. (Reddit's own `*.json` endpoints are now
    blocked for unauthenticated clients, so these mirrors are the no-account path.)
  - Archive full-text search is loose ("luce" = "light" in Italian), so a
    relevance filter (`config.RELEVANCE_ANY` + the `ferrari luce` pairing) drops
    off-topic hits; the collector logs `kept / fetched` so you can tune queries.
  - **Behind a proxy?** `requests` already honours the standard `HTTPS_PROXY` /
    `HTTP_PROXY` env vars, so `export HTTPS_PROXY=http://host:port` routes the
    archive calls through it — no code change. (Proxies are for network routing,
    *not* for evading Reddit's rate limits/blocks, which the ToS forbids.)

---

## 2. Run order

Each stage is a module under `src/` — run with `python -m src.<name>` (or call its
`main()` from the notebook).

| # | Command | Produces |
|---|---|---|---|
| 01 | `python -m src.collect_bluesky` | `posts_bluesky.csv`, `edges_follows.csv` |
| 02 | `python -m src.collect_reddit`| `reddit_submissions.csv`, `reddit_comments.csv` |
| 03 | `python -m src.build_graph` | `graph.graphml`, `nodes_centrality.csv`, `graph_summary.txt` |
| 04 | `python -m src.communities` | `nodes_communities.csv`, `communities_summary.txt` |
| 05 | `python -m src.content_sentiment` | `documents_sentiment.csv`, `aspect_sentiment.csv`, `sentiment_timeline.csv` |
| 06 | `python -m src.content_enrich`| `documents_enriched.csv`, `topics_keywords.csv`, `language_sentiment.csv` |
| 07 | `python -m src.ner_entities` | `entity_frequency.csv`, `entity_cooccurrence.csv`, `entity_sentiment.csv` |
| 08 | `python -m src.community_sentiment` | `community_sentiment.csv`, `community_camps.csv`, `documents_communities.csv` |
| 09 | `python -m src.viz` | all figures (`.png`) in `figures/` |

Stages 03–08 read the CSVs in `data/processed/`, so once collection (01–02) has
run you can re-run analysis freely. Outputs are written to `data/processed/`
(tables) and `figures/` 

---



## 3. Two preprocessing lanes
- **Lane A — word-level** (`utils.tokens_for_lexicon`, NLTK `TweetTokenizer`):
  feeds VADER/AFINN/NRC, word clouds. Heavier cleaning + lemmatisation.
- **Lane B — subword BPE** (`utils.clean_for_transformer` → model's
  `AutoTokenizer`): feeds twitter-roberta / irony / stance. Minimal cleaning
  (`@user`, `http`). The two are **not** chained.

---

## 4. Ethics & ToS
- Bluesky App Password + Reddit read-only OAuth only; no HTML scraping of Reddit.
- Only public posts; `.env` is git-ignored. Treat usernames as personal data —
  aggregate in the report, don't single out private individuals.


