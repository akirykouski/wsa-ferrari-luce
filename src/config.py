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
    '"Bad Bunny" "Super Bowl"',
    '"Bad Bunny" halftime',
    '"Super Bowl halftime" "Bad Bunny"',
    '"Super Bowl LX" "Bad Bunny"',
    '"Apple Music Halftime Show" "Bad Bunny"',
    "#BadBunny",
    "#SuperBowl",
    "#SuperBowlLX",
    "#HalftimeShow",
    '"Bad Bunny" "Lady Gaga"',
    '"Bad Bunny" "Ricky Martin"',
]

# REDDIT_QUERIES = [
#     "Ferrari Luce",
#     "Ferrari Elettrica",
#     "Ferrari electric",
#     "Ferrari EV",
#     "Ferrari electric car",
# ]

# Relevance filter for loose full-text matches (esp. PullPush). The car is
# "Ferrari Luce", but "luce" alone = "light" in Italian -> noise. We keep a post
# only if it has the exact "ferrari luce" pairing OR Ferrari + an EV-context token.
RELEVANCE_ANY = [
    "bad bunny",
    "benito",
    "halftime",
    "super bowl",
    "superbowl",
    "super bowl lx",
    "apple music halftime",
    "lady gaga",
    "ricky martin",
    "puerto rico",
    "spanish",
    "latino",
    "latin",
]

# SUBREDDITS = [
#     "Ferrari",
#     "cars",
#     "electricvehicles",
#     "formula1",
#     "stocks",
#     "wallstreetbets",
# ]

# Bluesky handles whose ego-network (followers/follows) we optionally crawl.
# Fill in real handles you discover during collection, e.g. "ferrari.com".
KEY_BLUESKY_ACCOUNTS: list[str] = []

# ----------------------------------------------------------------------------
# Time window (the Luce story)
# ----------------------------------------------------------------------------
# Capital Markets Day "Elettrica" preview -> production reveal -> stock drop.
SINCE_ISO = "2026-02-01T00:00:00Z"
FOCUS_SINCE_ISO = "2026-02-08T00:00:00Z"
UNTIL_ISO = "2026-02-20T00:00:00Z"
REDDIT_TIME_FILTER = "year"

EVENT_DATES = [
    ("2026-02-08", "Super Bowl LX halftime show"),
    ("2026-02-09", "Immediate review and reaction wave"),
    ("2026-02-13", "Post-show controversy / complaint coverage"),
]

# ----------------------------------------------------------------------------
# Aspect lexicons for Aspect-Based Sentiment Analysis (LAB 5, #CrazyPizza style)
# Each post is tagged with every aspect whose keywords it matches.
# ----------------------------------------------------------------------------
ASPECTS: dict[str, list[str]] = {
    "performance_energy": [
        "performance",
        "energy",
        "stage",
        "show",
        "live",
        "dance",
        "dancing",
        "choreography",
        "party",
        "boring",
        "amazing",
        "fire",
        "iconic",
    ],
    "music_setlist": [
        "song",
        "songs",
        "setlist",
        "medley",
        "album",
        "track",
        "tití",
        "monaco",
        "perreo",
        "apagón",
        "hits",
        "singing",
        "vocals",
    ],
    "spanish_language": [
        "spanish",
        "english",
        "language",
        "lyrics",
        "translate",
        "translation",
        "couldn't understand",
        "subtitles",
        "español",
    ],
    "latin_puerto_rican_identity": [
        "latin",
        "latino",
        "latina",
        "puerto rico",
        "puertorican",
        "boricua",
        "culture",
        "heritage",
        "representation",
        "pride",
        "island",
    ],
    "guest_appearances": [
        "lady gaga",
        "gaga",
        "ricky martin",
        "pedro pascal",
        "cardi b",
        "karol g",
        "guest",
        "cameo",
        "surprise",
    ],
    "visuals_production": [
        "visuals",
        "production",
        "lighting",
        "stage design",
        "costume",
        "outfit",
        "camera",
        "fireworks",
        "set",
        "dancers",
    ],
    "appropriateness_family": [
        "family",
        "kids",
        "children",
        "inappropriate",
        "sexual",
        "clean",
        "censored",
        "fcc",
        "complaint",
        "disgusting",
        "offensive",
    ],
    "super_bowl_fit": [
        "super bowl",
        "halftime",
        "nfl",
        "football",
        "american",
        "mainstream",
        "audience",
        "ratings",
        "commercial",
        "apple music",
    ],
}

# Comparison entities we expect to recur (used to sanity-check NER output)
WATCH_ENTITIES = [
    "Bad Bunny",
    "Benito",
    "Lady Gaga",
    "Ricky Martin",
    "Puerto Rico",
    "Apple Music",
    "NFL",
    "Super Bowl",
    "Levi's Stadium",
    "Roc Nation",
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
BLUESKY_PAGE_SIZE = 25  # API hard max for search_posts
REDDIT_MAX_PER_QUERY = 500
PULLPUSH_PAGE_SIZE = 100
