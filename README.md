# WSA Project — "Light or Letdown?": Social Media Analysis of the Ferrari Luce

Web and Social Media Search and Analysis (BSc AI, UniMiB). Track 1 — Social
Media Analysis: **Network Analysis + Content Analysis + Visualization** of the
online reaction to Ferrari's first electric car, the **Luce** (revealed 25 May 2026).

See [`MEGAPLAN.md`](MEGAPLAN.md) for the full objectives, research questions, and method design.

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

The pipeline lives entirely in `src/` (no separate `scripts/` wrappers); run each
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

| # | Command | Lab | Produces |
|---|---|---|---|
| 01 | `python -m src.collect_bluesky` | 1,2 | `posts_bluesky.csv`, `edges_follows.csv` |
| 02 | `python -m src.collect_reddit` | 1,2 | `reddit_submissions.csv`, `reddit_comments.csv` |
| 03 | `python -m src.build_graph` | 3 | `graph.graphml`, `nodes_centrality.csv`, `graph_summary.txt` |
| 04 | `python -m src.communities` | 4 | `nodes_communities.csv`, `communities_summary.txt` |
| 05 | `python -m src.content_sentiment` | 5 | `documents_sentiment.csv`, `aspect_sentiment.csv`, `sentiment_timeline.csv` |
| 06 | `python -m src.content_enrich` | + | `documents_enriched.csv`, `topics_keywords.csv`, `language_sentiment.csv` |
| 07 | `python -m src.ner_entities` | 6 | `entity_frequency.csv`, `entity_cooccurrence.csv`, `entity_sentiment.csv` |
| 08 | `python -m src.viz` | — | all figures (`.png`) in `figures/` |

Stages 03–08 read the CSVs in `data/processed/`, so once collection (01–02) has
run you can re-run analysis freely. Outputs are written to `data/processed/`
(tables) and `figures/` (all charts as `.png`, including the community-coloured
interaction network).

---

## 3. What maps to what (labs & research questions)

- **LAB 1** (scraping/`re`) → text cleaning in `utils.py` (no news scraped).
- **LAB 2** (atproto/PRAW) → `collect_bluesky.py`, `collect_reddit.py` (PRAW
  read-only primary; Arctic-Shift then PullPush.io as no-account fallbacks).
- **LAB 3** (centrality) → `build_graph.py` — directed interaction graph,
  in/out-degree, betweenness, closeness, PageRank, eigenvector (**RQ1**).
- **LAB 4** (communities) → `communities.py` — Louvain + greedy modularity,
  modularity Q, assortativity (**RQ2**).
- **LAB 5** (sentiment) → `content_sentiment.py` — VADER + AFINN + NRC emotions
  + twitter-roberta; aspect-based sentiment (**RQ3, RQ4**).
- **Enrichments** → `content_enrich.py` — sarcasm/irony (**RQ6**), stance
  (**RQ7**), topic modelling, language segmentation (**RQ8**).
- **LAB 6** (NER) → `ner_entities.py` — spaCy entities, co-occurrence,
  entity-level sentiment (**RQ5**).

**Graceful degradation:** if a transformer/topic model isn't installed, that
step logs a warning and is skipped while lexicon results (VADER/AFINN/NRC) and
all network analysis still complete.

---

## 4. Two preprocessing lanes (by design)
- **Lane A — word-level** (`utils.tokens_for_lexicon`, NLTK `TweetTokenizer`):
  feeds VADER/AFINN/NRC, word clouds. Heavier cleaning + lemmatisation.
- **Lane B — subword BPE** (`utils.clean_for_transformer` → model's
  `AutoTokenizer`): feeds twitter-roberta / irony / stance. Minimal cleaning
  (`@user`, `http`). The two are **not** chained.

---

## 5. Ethics & ToS
- Bluesky App Password + Reddit read-only OAuth only; no HTML scraping of Reddit.
- Only public posts; `.env` is git-ignored. Treat usernames as personal data —
  aggregate in the report, don't single out private individuals.

---

## 6. Submission checklist
- Rename this folder to **`WSA_surname1_surname2[_surname3]`** for the Google
  Drive submission.
- Include: this code + `README.md`, the `data/` you collected, the final report
  (`report/`), and the slides. Share ≥ 7 days before the exam with
  `marco.viviani@unimib.it`, `davide.mancino@unimib.it`, `m.braga@campus.unimib.it`.
