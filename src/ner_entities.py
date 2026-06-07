from __future__ import annotations
from collections import Counter, defaultdict
from itertools import combinations
import pandas as pd

from .utils import log, save_csv, load_csv
from .corpus import load_documents

KEEP_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "NORP", "LOC", "FAC", "WORK_OF_ART"}


def _load_base() -> pd.DataFrame:
    try:
        return load_csv("documents_sentiment.csv")
    except FileNotFoundError:
        return load_documents()


def _nlp():
    import spacy
    from . import config
    try:
        return spacy.load(config.SPACY_MODEL, disable=["lemmatizer"])
    except OSError as e:
        raise RuntimeError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        ) from e


def extract_entities(df: pd.DataFrame) -> dict[str, list[tuple[str, str]]]:
    nlp = _nlp()
    per_doc: dict[str, list[tuple[str, str]]] = {}
    texts = df["text"].fillna("").tolist()
    ids = df["doc_id"].tolist()
    for doc_id, doc in zip(ids, nlp.pipe(texts, batch_size=64)):
        ents = []
        for ent in doc.ents:
            txt = ent.text.strip()
            if ent.label_ in KEEP_LABELS and len(txt) > 2 and any(c.isalpha() for c in txt):
                ents.append((txt, ent.label_))
        per_doc[doc_id] = ents
    return per_doc


def frequency_table(per_doc) -> pd.DataFrame:
    """Document frequency (count each entity once per doc) so a single spammy
    post that repeats a token 100x can't dominate the ranking."""
    counter = Counter()
    labels = {}
    for ents in per_doc.values():
        seen = {}
        for text, lab in ents:
            seen.setdefault(text.lower(), lab)
        for key, lab in seen.items():
            counter[key] += 1
            labels.setdefault(key, lab)
    rows = [{"entity": k, "label": labels[k], "count": c}
            for k, c in counter.most_common()]
    df = pd.DataFrame(rows)
    save_csv(df, "entity_frequency.csv")
    return df


def cooccurrence(per_doc, min_count: int = 2, max_per_doc: int = 25) -> pd.DataFrame:
    """Cap entities per document before pairing, so one entity-dense (often
    spammy) doc can't blow up the combinatorics into hundreds of thousands of edges."""
    pair = Counter()
    for ents in per_doc.values():
        uniq = sorted({t.lower() for t, _ in ents})[:max_per_doc]
        for a, b in combinations(uniq, 2):
            pair[(a, b)] += 1
    rows = [{"source": a, "target": b, "weight": w}
            for (a, b), w in pair.items() if w >= min_count]
    df = pd.DataFrame(rows).sort_values("weight", ascending=False) if rows else pd.DataFrame(
        columns=["source", "target", "weight"])
    save_csv(df, "entity_cooccurrence.csv")
    return df


def entity_sentiment(df: pd.DataFrame, per_doc) -> pd.DataFrame:
    label_col = next((c for c in ("sentiment_corrected", "transformer_label", "vader_label")
                      if c in df.columns), None)
    if not label_col:
        log.warning("no sentiment column; skipping entity sentiment.")
        return pd.DataFrame()
    doc_label = dict(zip(df["doc_id"], df[label_col]))
    agg = defaultdict(Counter)
    for doc_id, ents in per_doc.items():
        lab = doc_label.get(doc_id)
        if lab is None:
            continue
        for text, _ in {(t.lower(), l) for t, l in ents}:
            agg[text][str(lab)] += 1
    rows = []
    for ent, c in agg.items():
        total = sum(c.values())
        rows.append({
            "entity": ent, "total": total,
            "positive": c.get("positive", 0),
            "neutral": c.get("neutral", 0),
            "negative": c.get("negative", 0),
            "negative_ratio": round(c.get("negative", 0) / total, 3) if total else 0,
        })
    out = pd.DataFrame(rows).sort_values("total", ascending=False)
    save_csv(out, "entity_sentiment.csv")
    return out


def main() -> None:
    df = _load_base()
    if df.empty:
        log.warning("empty corpus; run collectors first.")
        return
    per_doc = extract_entities(df)
    freq = frequency_table(per_doc)
    cooccurrence(per_doc)
    entity_sentiment(df, per_doc)
    log.info("Top entities:\n%s", freq.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
