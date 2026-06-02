"""Central configuration: paths, queries, aspect lexicons, models, event dates.

Everything tunable lives here so the notebooks/scripts stay thin.
"""
from __future__ import annotations
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
FIGURES = BASE_DIR / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------------
# Topic definition: queries / subreddits / accounts
# ----------------------------------------------------------------------------
# Bluesky search supports Boolean ops:  space=AND, "|"=OR, "-"=NOT, quotes=phrase
BLUESKY_QUERIES = [
    '"Ferrari Luce"',
    "Ferrari Luce",
    '"Ferrari EV"',
    "Ferrari electric",
    "Ferrari Elettrica",
    "#FerrariLuce",
    "Ferrari Luce ugly",          # surface the backlash explicitly
]

REDDIT_QUERIES = [
    "Ferrari Luce",
    "Ferrari Elettrica",
    "Ferrari electric",
    "Ferrari EV",
    "Ferrari electric car",
]

# Relevance filter for loose full-text matches (esp. PullPush). The car is
# "Ferrari Luce", but "luce" alone = "light" in Italian -> noise. We keep a post
# only if it has the exact "ferrari luce" pairing OR Ferrari + an EV-context token.
RELEVANCE_ANY = [
    "electric", "elettrica", " ev", "ev ", "battery", "all-electric",
    "jony ive", "lovefrom", "first electric", "maranello ev",
]

SUBREDDITS = [
    "Ferrari",
    "cars",
    "electricvehicles",
    "formula1",
    "stocks",
    "wallstreetbets",
]

# Bluesky handles whose ego-network (followers/follows) we optionally crawl.
# Fill in real handles you discover during collection, e.g. "ferrari.com".
KEY_BLUESKY_ACCOUNTS: list[str] = []

# ----------------------------------------------------------------------------
# Time window (the Luce story)
# ----------------------------------------------------------------------------
# Capital Markets Day "Elettrica" preview -> production reveal -> stock drop.
SINCE_ISO = "2025-10-01T00:00:00Z"     # widest window (since CMD)
FOCUS_SINCE_ISO = "2026-05-20T00:00:00Z"  # the dense reveal window
UNTIL_ISO = None                        # None = up to now
REDDIT_TIME_FILTER = "year"             # PRAW search: all/year/month/week/day

# Event markers for the temporal-sentiment chart (annotated manually, not scraped)
EVENT_DATES = [
    ("2025-10-09", "Capital Markets Day (Elettrica preview)"),
    ("2026-05-25", "Luce reveal near Rome"),
    ("2026-05-26", "Stock falls ~8%"),
]

# ----------------------------------------------------------------------------
# Aspect lexicons for Aspect-Based Sentiment Analysis (LAB 5, #CrazyPizza style)
# Each post is tagged with every aspect whose keywords it matches.
# ----------------------------------------------------------------------------
ASPECTS: dict[str, list[str]] = {
    "exterior_look": [
        "design", "look", "looks", "styling", "ugly", "beautiful", "shape",
        "proportions", "silhouette", "aesthetic", "hideous", "gorgeous",
        "bubble", "blob", "egg", "front", "rear",
    ],
    "interior": [
        "interior", "cabin", "seats", "seat", "five-seater", "5-seater",
        "five seater", "practical", "space", "dashboard", "screen", "back seat",
    ],
    "sound": [
        "sound", "noise", "engine note", "exhaust", "silent", "vibration",
        "artificial sound", "fake sound", "soundtrack", "roar", "audio",
    ],
    "price": [
        "price", "expensive", "cost", "euro", "550", "640", "money", "value",
        "overpriced", "cheap", "worth", "afford", "$",
    ],
    "performance": [
        "performance", "power", "fast", "acceleration", "0-100", "0 to 60",
        "top speed", "range", "horsepower", "battery", "torque", "quick", "cv",
    ],
    "name": ["name", "luce", "called", "naming", "named"],
    "electrification": [
        "electric", "ev", "battery", "charging", "charge", "combustion",
        "ice", "petrol", "gas", "hybrid", "electrify", "electrification", "evs",
    ],
    "brand_identity": [
        "heritage", "soul", "real ferrari", "not a ferrari", "tradition",
        "prancing horse", "brand", "identity", "betrayal", "maranello",
        "sellout", "legacy",
    ],
    "design_house": [
        "jony ive", "ive", "apple", "lovefrom", "designer", "iphone", "tim cook",
    ],
}

# Comparison entities we expect to recur (used to sanity-check NER output)
WATCH_ENTITIES = [
    "Nissan Leaf", "Tesla", "Apple", "Jony Ive", "LoveFrom", "Benedetto Vigna",
    "NIO", "Maranello", "Ferrari",
]

# Languages of analytical interest (RQ8: Italian national-pride angle)
LANGUAGES = ["en", "it"]

# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
IRONY_MODEL = "cardiffnlp/twitter-roberta-base-irony"
OFFENSIVE_MODEL = "cardiffnlp/twitter-roberta-base-offensive"
STANCE_NLI_MODEL = "facebook/bart-large-mnli"
SPACY_MODEL = "en_core_web_sm"

# Stance targets (zero-shot NLI). Each maps to a hypothesis template.
STANCE_TARGETS = {
    "luce": "This text is in favor of the Ferrari Luce car.",
    "ev_transition": "This text is in favor of electric vehicles.",
}

# ----------------------------------------------------------------------------
# Collection caps (be polite to the APIs)
# ----------------------------------------------------------------------------
BLUESKY_MAX_PER_QUERY = 1000
BLUESKY_PAGE_SIZE = 25          # API hard max for search_posts
REDDIT_MAX_PER_QUERY = 500
PULLPUSH_PAGE_SIZE = 100
