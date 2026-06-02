# WSA PROJECT MEGAPLAN
## "Light or Letdown? A Web & Social Media Analysis of the Ferrari Luce Launch"

**Course:** Web and Social Media Search and Analysis — BSc Artificial Intelligence, University of Milano-Bicocca (Prof. Marco Viviani)
**Track chosen:** Track 1 — *Social Media Analysis* (Network Analysis + Content Analysis + Visualization). **Scope locked to the core (Labs 1–6).** The Social Search + RAG extension (Labs 7–8) is documented in §3C as *optional future work*, not part of this project.
**Data sources locked:** **Bluesky (atproto) + Reddit (PRAW)** only. No news/web data is collected; Lab 1's `re`/BeautifulSoup techniques are reused for cleaning/parsing the social text.
**Date of plan:** 2026-06-02

---

## 0. WHY THIS TOPIC

The **Ferrari Luce** (Type F222) is Ferrari's first production battery-electric vehicle and its first five-seater. Key facts:
- Previewed as "Elettrica" at Capital Markets Day (9 Oct 2025); production car revealed near Rome on **25 May 2026**.
- Designed in collaboration with **Jony Ive's LoveFrom**; four in-wheel motors (~1,050 cv), 0–100 in 2.5 s, ~530 km range.
- Priced from **€550,000** (~$640k); deliveries Q4 2026.
- **Reception was explosively negative**: fans called it "insulting," likened it to a Nissan Leaf, Italy's transport minister mocked the styling, and **Ferrari stock fell ~8%**. Some analysts (and NIO's CEO) defended it.

**Why it is an ideal WSA topic:**
1. **Fresh + bounded** — the reveal is ~1 week old, so we get a clean, dense, datable event window (CMD Oct'25 → reveal 25 May'26 → stock crash → ongoing).
2. **Highly polarizing** — strong, emotional opinions = rich sentiment/emotion signal (not lukewarm).
3. **Multi-community** — Ferrari purists, EV enthusiasts, Apple/design admirers, F1 fans, finance/investor crowd ($RACE), Italian national-pride angle → ideal for community detection.
4. **Entity-dense** — Jony Ive, Apple, LoveFrom, CEO (Benedetto Vigna), Tesla, NIO, Nissan Leaf, Italian transport minister → ideal for NER + entity-level sentiment.
5. **Mirrors the professor's own examples** — the brief literally shows "F1 Community Analysis (Ferrari)," #FlatEarth word clouds, and #CrazyPizza aspect-sentiment; we hit all three patterns on a hotter, newer event.

---

## 1. OBJECTIVE  *(Rubric: 1 pt — completeness & clarity)*

**Phenomenon to describe:** How the global online public reacted to the Ferrari Luce launch — *who* drove the conversation, *which communities* formed, *what* they felt, *which aspects/entities* drove the backlash, and how this evolved around the stock crash.

**Research questions:**
- **RQ1 (Network):** Who are the most central accounts (influencers, brokers, bridges)? Is the conversation a single hub or fragmented hubs?
- **RQ2 (Community):** What discourse communities exist, how cohesive are they (modularity), and do pro/anti camps interconnect or form echo chambers (assortativity)?
- **RQ3 (Sentiment & Emotion):** What is the overall polarity and emotion mix? How does sentiment differ **by community** and **over time** (reveal → stock drop)?
- **RQ4 (Aspect):** Which product aspects (exterior look, interior, **sound/audio**, price/value, performance, the name "Luce", electrification, brand heritage, the Jony Ive/Apple influence) drive the negativity?
- **RQ5 (Entities):** Which entities and comparisons recur (Nissan Leaf, Tesla, Apple, Vigna, transport minister), and what is the sentiment *toward each*?
- **RQ6 (Sarcasm):** How prevalent is sarcasm/irony in the backlash, and how much does it distort naïve sentiment scores?
- **RQ7 (Stance):** Independent of tone, is each post *for* or *against* the Luce — and, separately, for/against the EV transition? (A user can love EVs yet reject this car.)
- **RQ8 (Language/identity):** Does sentiment/stance differ by post language (Italian vs English vs other), testing the "wounded national pride" angle?

---

## 2. DATA COLLECTION  *(Rubric: 1 pt — correctness)* — **uses LAB 1 & LAB 2**

### Sources & methods
| Source | Lab technique | What we collect |
|---|---|---|
| **Bluesky** | LAB 2 — `atproto` `app.bsky.feed.search_posts` (q, sort, limit≤25, cursor, since/until) | Posts matching queries; text, `created_at`, author (handle/did/display_name), engagement (`like_count`, `repost_count`, `reply_count`, `quote_count`), facets (**hashtags, mentions, links**), `langs`, `reply_parent_uri`/`reply_root_uri` for threads |
| **Bluesky graph** | LAB 2 — `get_followers` / `get_follows` (paginated) | Ego-networks of key accounts (@ferrari, motoring journalists, finance accounts) for follower-based network |
| **Reddit** | LAB 2 — `praw` `subreddit().search(...)`, `.new/.hot/.top`, comment trees | Submissions + comments from r/Ferrari, r/cars, r/electricvehicles, r/formula1, r/stocks, r/wallstreetbets ($RACE) |

**No web/news data is collected** (per scope decision — news-article comment sections are explicitly excluded). Key public event dates (CMD 9 Oct 2025, reveal 25 May 2026, stock drop 26 May 2026) are annotated **manually** on the temporal charts — no scraping needed. Lab 1's `re`/BeautifulSoup skills are instead reused to **clean/parse the social text**.

### Reddit collection strategy (PRAW primary + PullPush fallback) — *locked*
- **Primary — PRAW, read-only OAuth** (`client_id` + `client_secret` + `user_agent`, **no password**): the method taught in Lab 2 and the only sanctioned, ToS-compliant route. Free non-commercial tier (~100 QPM per OAuth client; PRAW self-throttles). Boolean `search()` across **r/Ferrari, r/cars, r/electricvehicles, r/formula1, r/stocks, r/wallstreetbets** with `time_filter`.
- **Comment trees:** always call `submission.comments.replace_more(limit=0)` before flattening, otherwise replies (our reply-edges) are silently dropped.
- **1000-item listing cap:** every listing (`search`/`new`/`top`/`hot`) returns ≤ ~1000 items and PRAW search is recency/relevance-biased; we diversify across subreddits × sort × `time_filter` × keywords to stay under it (the post-25-May window is small enough that this rarely binds).
- **Fallback — PullPush.io** (free, no API key; limits 15 soft / 30 hard rpm, 1000/hour) for time-sliced historical pulls only if a PRAW query saturates the cap or we want deeper comment history.
- **Never** HTML-scrape old.reddit / `.json` endpoints (blocked + against ToS).

### Query design
Boolean queries (Bluesky supports `OR` via `|`, `-` for NOT): `"Ferrari Luce"`, `"Ferrari EV"`, `"Ferrari Elettrica"`, `#FerrariLuce`, `$RACE Ferrari`, `Ferrari electric`, plus Italian terms (`Ferrari elettrica`, `Ferrari Luce`). Time window: **2025-10-01 → present**, with focus on **2026-05-20 → present**.

### Hygiene & ethics
- Dedup by URI/permalink; language tag filter (EN + IT); simple bot heuristics; keep **raw JSON + cleaned CSV**.
- Document every column (data dictionary in README); record collection date & query strings for reproducibility.
- Respect ToS / rate limits → reusable **retry+exponential-backoff** wrapper (LAB 2) and local caching. Store secrets in `.env` (Bluesky app password, Reddit client id/secret).

### Output datasets
`posts.csv`, `comments.csv`, `nodes.csv`, `edges_mentions.csv`, `edges_replies.csv`, `edges_follows.csv`. (Event dates for the timeline are a small hand-written constants list, not a scraped file.)

---

## 3. MODELS / METHODS  *(Rubric: 2.5 pts — correctness)*

> Track 1 requires **BOTH** Network Analysis **AND** Content Analysis. We do both, then add the Search/RAG extension.

### 3A. NETWORK ANALYSIS — **LAB 3 + LAB 4**
**Graph construction (LAB 3):** primary graph = **directed**, weighted interaction graph. Nodes = accounts; edges = mentions / replies / reposts / quotes (weight = interaction count). Secondary follower graph from `get_follows`.

> **Why directed (and when undirected)?** These interactions are inherently *asymmetric*: "A replies to / mentions / reposts / follows B" ≠ the reverse. Direction is what separates **in-degree = attention received** (being talked about → influence) from **out-degree = activity** (doing the talking), and it's what makes PageRank/eigenvector and directed betweenness meaningful for RQ1. An undirected graph collapses a *beloved hub everyone mentions* and a *spammer who mentions everyone* into the same "high degree," destroying exactly the signal we want. **So: directed for centrality/influence.** For **community detection (Lab 4)** we **symmetrize to undirected** (optionally weighted), since Louvain/greedy-modularity were taught and are standard on undirected graphs (Karate Club, Facebook) and "who clusters with whom" needs no direction. Genuinely symmetric relations (mutual follows, hashtag co-usage) are also modeled undirected. **Net rule: directed for influence, undirected for communities.**

**Centrality & structure (LAB 3):**
- `degree_centrality` (in/out → most active / most addressed), `betweenness_centrality` (brokers bridging camps), `closeness_centrality`, **PageRank/eigenvector** (influence).
- Global metrics: density, **transitivity/clustering coefficient**, connected components, diameter/radius, degree distribution (hub detection) — note: the brief's F1 example compares density vs transitivity, so we replicate that chart.

**Community detection (LAB 4):**
- `louvain_communities` + `greedy_modularity_communities` (compare partitions), report **modularity Q**; `girvan_newman` on a focused subgraph for illustration; optionally `asyn_fluidc`.
- **Assortativity** to test echo-chamber hypothesis; identify **crossing edges** between communities.
- Label each community by inspecting top members + dominant hashtags + dominant sentiment (links to 3B).

**Network visualization:** `spring_layout` colored by community, node size by centrality; export to **Gephi/pyvis** for the high-impact "hairball" (cf. the F1-Ferrari slide).

### 3B. CONTENT ANALYSIS — **LAB 5 + LAB 6**
**Preprocessing — two parallel tokenization lanes (TweetTokenizer vs BPE):**
- **Lane A — word-level (`nltk.TweetTokenizer`)** feeds the *lexicon & classical* methods (VADER, AFINN, NRCLex, TF-IDF/CountVectorizer, word clouds). These score *words*, so we want readable word tokens: TweetTokenizer keeps #hashtags/@mentions/emoticons, shortens elongations (`looove`→`loove`); we add Lab 1 `re` cleaning (normalise URLs), lowercasing, stopword removal, lemmatization.
- **Lane B — subword byte-level BPE** is the transformer's *own* tokenizer (`AutoTokenizer`) for twitter-roberta sentiment / irony / stance. We feed **near-raw** text with only cardiffnlp normalisation (`@user`, `http`) and **skip** stopword removal/lemmatization (they hurt a model trained on natural tweets).
- **Not stackable in sequence:** you must *not* BPE the output of TweetTokenizer — double-tokenizing breaks the model's token↔id alignment. The two are parallel lanes selected by the downstream model; the project uses **both**, one per model family.

**Sentiment (LAB 5):**
- **Lexicon baselines:** VADER (`SentimentIntensityAnalyzer`, compound) + AFINN.
- **Primary classifier:** transformer **`cardiffnlp/twitter-roberta-base-sentiment`** (pos/neu/neg) via HuggingFace.
- (Optional) `sklearn` LogisticRegression + CountVectorizer baseline on a hand-labeled sample → report accuracy / confusion matrix / P-R-F (LAB 5's evaluation pattern).
- **Method agreement** analysis; **temporal sentiment timeline** (reveal → stock drop) with post-volume overlay (cf. F1 example dual-axis chart); **per-community sentiment** (join with 3A).

**Emotion (LAB 5 — `NRCLex`):** emotion profile (anger, disgust, sadness, anticipation, joy, trust…) — quantify "brand-betrayal" emotions.

**Aspect-Based Sentiment (the #CrazyPizza pattern):** assign each post to one or more **product aspects**, then compute pos/neg/neu per aspect → aspect × sentiment **table + stacked bar + pie** (mirrors the brief's example). Aspects:
- **Exterior look / design** — the headline controversy
- **Interior / packaging** — 5-seater, practicality
- **Sound / audio** — the *artificial engine vibration & sound*, a uniquely contentious aspect for an electric Ferrari
- **Price / value** — €550k
- **Performance** — 1,050 cv, 0–100 in 2.5 s, range
- **Name** — "Luce"
- **Electrification / EV-ness** — whether a Ferrari "should" be electric
- **Brand identity / heritage** — "not a real Ferrari"
- **Jony Ive / Apple influence** — the design-house angle

Aspects are seeded with keyword lexicons and **validated/extended by data-driven topic modelling** (below) so we don't miss aspects we didn't anticipate.

> **Is "language" an aspect?** No — language is a property of the *post*, not the *car*, so it is **not** in the aspect list. But it **is** an important cross-cutting **segmentation dimension**: given the strong Italian-national-pride angle (transport minister, domestic fans), comparing sentiment/stance/aspects **Italian vs English vs other** is analytically valuable. We treat language as a *grouping variable* across RQ3–RQ7, taken from Bluesky's `langs` field and `langdetect` for Reddit (→ **RQ8**).

**Content-analysis enrichments (build on Lab 5's transformer approach):**
- **Sarcasm / irony detection (NEW):** the backlash is heavily ironic ("a €550k Nissan Leaf"), which *flips* lexical sentiment and corrupts naïve scores. We flag irony with **`cardiffnlp/twitter-roberta-base-irony`** (same twitter-roberta family as our sentiment model; SemEval-2018 Task 3) to (a) quantify sarcasm prevalence, (b) **re-interpret sentiment** on flagged posts (an "ironically positive" post is really negative), (c) compare sarcasm rates across communities. → **RQ6**
- **Stance detection (extension #1):** sentiment ≠ stance. We classify each post's **stance toward the Luce** (favor / against / none) and, separately, **toward EVs in general**, isolating "I love EVs but this is ugly." Zero-/few-shot via an NLI model (e.g., `facebook/bart-large-mnli`) with target hypotheses, validated on a hand-labelled sample. → **RQ7**
- **Topic modelling (extension #2):** discover themes **bottom-up** with **BERTopic** (embeddings + clustering) and/or **LDA (gensim)**, then map topics back onto the hand-built aspects — this *validates* the ABSA aspect list and surfaces unanticipated themes (charging, resale value, F1 brand spillover). Adds an unsupervised method alongside the supervised ones (strengthens the "Models" rubric).
- *(Optional)* **Toxicity/offensiveness** (`cardiffnlp/twitter-roberta-base-offensive`) to quantify how hostile the discourse turned — natural given the "insulting" framing.

**NER (LAB 6):** spaCy `en_core_web_sm` (PERSON/ORG/GPE/PRODUCT) with `nltk.ne_chunk` comparison; `Counter.most_common` frequency; **entity linking** via DBpedia Spotlight / Babelfy; **entity co-occurrence network**; **entity-level sentiment** (sentiment toward "Nissan Leaf", "Jony Ive", "Vigna", "Tesla").

**Word clouds (LAB 2/3):** per-community and per-sentiment clouds (cf. #FlatEarth slide).

### 3C. OPTIONAL FUTURE WORK (OUT OF SCOPE) — SOCIAL SEARCH + RAG — **LAB 7 + LAB 8**
> Not part of this project. Documented only so the path to an all-labs "mega" version is on record if scope is ever expanded.
- **Social search (LAB 7, PyTerrier):** index posts; BM25/TF-IDF baseline; **re-rank with social signals** — engagement, **author centrality from 3A**, sentiment, recency, credibility = multi-dimensional relevance. Hand-built **qrels**; evaluate **P@k, nDCG@k, MAP, R-prec**.
- **RAG QA (LAB 8, pyterrier_rag):** corpus = posts + headlines. Pipeline **BM25 → MonoT5 rerank → reader** (FlanT5 / T5-FiD), report **EM/F1**.

---

## 4. VISUALIZATION PLAN  *(Rubric: 1 pt — effectiveness)*
- Community-colored network hairball (Gephi/pyvis), node size = centrality.
- Centrality leaderboard table + bar charts; density-vs-transitivity bar (F1 style).
- Modularity comparison across algorithms.
- Sentiment distribution (bar/pie); **NRC emotion bar/radar**; **sentiment-over-time** dual-axis with volume.
- **Aspect × sentiment** stacked bar + pie (#CrazyPizza style).
- Entity-frequency bars + entity co-occurrence graph.
- Word clouds per community / per sentiment.
- Search: metric tables; RAG: example "answer cards."

---

## 5. LAB → PROJECT TRACEABILITY  *(proves we use what we learned)*
| Lab | Technique | Used in | In scope? |
|---|---|---|---|
| **LAB 1** Web Scraping | BeautifulSoup, **`re`**, pandas | `re`-based cleaning/parsing of social text in preprocessing (no news corpus collected) | Techniques reused |
| **LAB 2** Bluesky/Reddit | atproto, praw, retry/backoff, facets | **Primary data collection** + graph edges (§2) | ✅ Core |
| **LAB 3** SNA | networkx centralities, layouts | Influencer/broker analysis, structure (§3A) | ✅ Core |
| **LAB 4** Community Detection | Louvain, greedy modularity, Girvan-Newman, modularity, assortativity | Discourse communities, echo chambers (§3A) | ✅ Core |
| **LAB 5** Sentiment | VADER, AFINN, NRCLex, twitter-roberta, sklearn eval | Sentiment, emotion, aspect-based, timeline (§3B) | ✅ Core |
| **LAB 6** NER | spaCy, nltk ne_chunk, DBpedia/Babelfy linking | Entity extraction, linking, entity sentiment (§3B) | ✅ Core |
| **LAB 7** IR/Social Search | PyTerrier BM25/TF-IDF, P@k/nDCG/MAP, social features | (§3C) | Out of scope |
| **LAB 8** RAG/Gen-IR | BM25→MonoT5→FlanT5/FiD, doc2query, E5, EM/F1 | (§3C) | Out of scope |

*Beyond-the-labs content enrichments (built on Lab 5's transformer pipeline): sarcasm/irony (`twitter-roberta-base-irony`, RQ6), stance (NLI zero-shot, RQ7), topic modelling (BERTopic/LDA), language segmentation (RQ8).*

---

## 6. REPOSITORY & REPRODUCIBILITY
```
WSA_<surname1>_<surname2>/
  README.md                # how to run, data dictionary, secrets setup
  requirements.txt
  .env.example             # BLUESKY_HANDLE/APP_PASSWORD, REDDIT creds
  data/ raw/  processed/
  notebooks/
    01_collect_bluesky.ipynb        # LAB 2  (core)
    02_collect_reddit.ipynb         # LAB 2  (core)
    03_build_graph.ipynb            # LAB 3  (core)
    04_communities.ipynb            # LAB 4  (core)
    05_sentiment_emotion.ipynb      # LAB 5  (core)
    06_sarcasm_stance_topics.ipynb  # enrichments (irony, stance, topic modelling, language)
    07_ner_entities.ipynb           # LAB 6  (core)
  src/                     # shared helpers (collectors, cleaning, viz)
  figures/                 # exported charts for report & slides
  report/                  # final report (PDF) + slides
```
**Core deps:** `atproto, praw, requests, beautifulsoup4, pandas, numpy, networkx, python-louvain, nltk, vaderSentiment, afinn, nrclex, transformers, torch, spacy (+en_core_web_sm), langdetect, bertopic, gensim, wordcloud, matplotlib, seaborn, plotly, pyvis, scikit-learn, python-dotenv`. *(PyTerrier / pyterrier_t5 / pyterrier_rag / sentence-transformers only needed if §3C is ever pursued.)*

---

## 7. DELIVERABLES & SUBMISSION  *(per project PDF)*
- **Final report** structured exactly as the rubric: **Objective / Data / Models / Results (+ visualization & discussion)**.
- **Data** folder + **source code with README** (the repo above).
- **~15-min PowerPoint**; **each member presents an aspect**. Suggested split:
  - Member A: Objective + Data collection (Labs 1–2) + Network analysis (Lab 3) + community detection (Lab 4).
  - Member B: Sentiment/emotion/aspect-based analysis (Lab 5) + NER & entity sentiment (Lab 6).
  - Member C *(if 3; if 2, split B/C)*: Enrichments — sarcasm, stance, topic modelling, language segmentation — + visualization/synthesis.
- **Submission:** Google Drive folder named **`WSA_surname1_surname2_...`**, shared to `marco.viviani@unimib.it`, `davide.mancino@unimib.it`, `m.braga@campus.unimib.it`, **≥7 days before the written exam**, in association with the current exam.

---

## 8. TIMELINE (suggested, ~3–4 weeks, core scope)
- **Week 1 — Scope & Data:** finalize RQs; build collectors (Labs 1–2, Bluesky + Reddit); collect & clean; freeze datasets + data dictionary.
- **Week 2 — Network:** graph build, centralities (Lab 3), communities + modularity/assortativity (Lab 4); Gephi export.
- **Week 3 — Content:** sentiment + emotion + aspect (Lab 5), sarcasm/stance/topic-modelling enrichments + language segmentation, NER + entity sentiment (Lab 6); join sentiment↔community↔time↔language.
- **Week 4 — Write-up:** finalize all figures; write report; build slides; submit.

---

## 9. RISKS & MITIGATIONS
- **Sparse Bluesky volume** for a luxury topic → add Reddit + scraped comments + **Italian-language** posts; widen queries/window.
- **API rate limits** → retry/backoff (Lab 2) + caching.
- **Sentiment domain mismatch** → twitter-roberta primary + hand-validated sample + lexicon cross-check.
- **Graph too dense/sparse** → edge-weight thresholding, focus on giant component.
- **RAG hallucination** → ground strictly on retrieved passages; report retrieval metrics; keep extension clearly scoped.

---

## 10. EXPECTED HEADLINE FINDINGS (hypotheses to test)
- Overwhelmingly **negative** sentiment dominated by **anger/disgust** around **design** and **brand identity**, with a secondary **price** axis.
- Distinct communities: *purists/anti*, *EV-optimists/defenders*, *finance/$RACE*, *design-Apple admirers*, *meme/comedy* — with **low cross-camp assortativity** (echo chambers).
- A handful of **journalist/influencer hubs** seed most reposts; the stock-crash event spikes both **volume** and **negativity**.
- **High sarcasm/irony rate** → naïve sentiment *understates* the negativity; irony-correction pushes more posts negative (RQ6).
- **Stance ≠ sentiment:** a meaningful share of EV-positive users are nonetheless *anti-Luce* (RQ7).
- **Italian-language** posts skew harsher on **heritage/identity & the Jony-Ive/Apple angle** than English ones (RQ8).
