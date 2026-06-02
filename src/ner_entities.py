"""LAB 6 — Named Entity Recognition + entity-level sentiment.

spaCy (en_core_web_sm) extracts entities; we report frequency, an entity
co-occurrence network, and—joined with the sentiment layer—the sentiment
distribution *toward* each entity (e.g. "Nissan Leaf", "Jony Ive", "Vigna").

Run:  python -m src.ner_entities   (best after content_sentiment)
Outputs: entity_frequency.csv , entity_cooccurrence.csv , entity_sentiment.csv
"""
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
            if ent.label_ in KEEP_LABELS and len(ent.text.strip()) > 1:
                ents.append((ent.text.strip(), ent.label_))
        per_doc[doc_id] = ents
    return per_doc


def frequency_table(per_doc) -> pd.DataFrame:
    counter = Counter()
    labels = {}
    for ents in per_doc.values():
        for text, lab in ents:
            key = text.lower()
            counter[key] += 1
            labels.setdefault(key, lab)
    rows = [{"entity": k, "label": labels[k], "count": c}
            for k, c in counter.most_common()]
    df = pd.DataFrame(rows)
    save_csv(df, "entity_frequency.csv")
    return df


def cooccurrence(per_doc, min_count: int = 2) -> pd.DataFrame:
    pair = Counter()
    for ents in per_doc.values():
        uniq = sorted({t.lower() for t, _ in ents})
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
