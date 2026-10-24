"""Shared paths, text repair, and fact extraction for the HoloMine price task.

Every EDA and preprocessing script imports from here so that a change to the
cleaning rules propagates to the analysis and to the training notebook at once.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
PREP = ROOT / "prep_out"

SEED = 42
PRICE_FLOOR = 10_000.0

for _d in (FIGURES, TABLES, PREP):
    _d.mkdir(parents=True, exist_ok=True)


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    train["text"] = train["text"].fillna("")
    test["text"] = test["text"].fillna("")
    return train, test


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLES / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_fig(fig, name: str) -> Path:
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    return path


SPLIT_NUMBER = re.compile(r"(?<=\d),\s+(?=\d{3}\b)")
GLUED_SENTENCE = re.compile(r"(?<=[a-z0-9])\.(?=[A-Z])")
GLUED_PUNCT = re.compile(r"(?<=[a-z])([,;:!?])(?=[A-Za-z])")
MULTISPACE = re.compile(r"\s+")


def repair_text(text: str) -> str:
    """Undo the two corruptions that the scraper baked into every split.

    1. Thousands separators were turned into ", " so "12, 400 square feet"
       is really 12400 square feet. Left alone, every number parser and every
       tokenizer reads it as two separate numbers.
    2. Punctuation lost its trailing space ("convenience.Step", "views,this"),
       which glues two real words into one out-of-vocabulary token. The period
       rule only fires before a capital so that decimals and "U.S.A" survive;
       the lookbehind on a lowercase letter keeps thousands separators safe.
    """
    text = SPLIT_NUMBER.sub("", text)
    text = GLUED_SENTENCE.sub(". ", text)
    text = GLUED_PUNCT.sub(r"\1 ", text)
    text = MULTISPACE.sub(" ", text)
    return text.strip()


NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"

MOD = r"(?:\+/-|\+\-)?[\s-]*(?:(?:deeded|irrigated|usable|buildable|wooded|level|flat|total|gross|open|pasture|tillable|more|or|less|approx\.?|approximately|additional|private|beautiful|rolling)[\s-]+){0,3}"

FACT_PATTERNS: dict[str, str] = {
    "sqft": rf"{NUM}[\s-]*{MOD}(?:square[-\s]?(?:feet|foot|ft)|sq\.?\s?ft\.?|sf\b|ft2\b)",
    "acres": rf"{NUM}[\s-]*{MOD}(?:acres?\b|ac\.?\s)",
    "beds": rf"{NUM}[\s-]*(?:bed\b|beds\b|bedroom|br\b|bdrm|bd\b)",
    "baths": rf"{NUM}[\s-]*(?:bath|ba\b|full bath)",
    "car_garage": rf"{NUM}[\s-]*car\b",
    "stories": rf"{NUM}[\s-]*(?:story|stories)\b",
    "units": rf"{NUM}[\s-]*unit",
}
YEAR_PATTERN = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")
BUILT_YEAR = re.compile(
    r"(?:built|constructed|erected|established|circa|c\.)\s*(?:in\s*)?(1[89]\d{2}|20[0-2]\d)"
    r"|(1[89]\d{2}|20[0-2]\d)[\s-]*(?:built|construction|vintage|colonial|bungalow|farmhouse|victorian)",
    re.I,
)
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
WORD_BED = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")[\s-]+bed", re.I)
WORD_BATH = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")[\s-]+(?:and[\s-]+a[\s-]+half[\s-]+)?bath", re.I)
WORD_HALF_BATH = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")[\s-]+and[\s-]+a[\s-]+half[\s-]+bath", re.I)

SANE_RANGE = {
    "sqft": (120.0, 200_000.0),
    "acres": (0.005, 200_000.0),
    "beds": (1.0, 40.0),
    "baths": (0.5, 40.0),
    "car_garage": (1.0, 12.0),
    "stories": (1.0, 6.0),
    "units": (1.0, 500.0),
}


def _to_float(token: str) -> float:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return np.nan


def extract_facts(text: str) -> dict[str, float]:
    """Pull the structured property facts back out of the free text.

    Values outside a plausible physical range are dropped rather than clipped,
    because a bad parse ("1 of 3 bathrooms") is noise, not a small house.
    """
    out: dict[str, float] = {}
    for name, pattern in FACT_PATTERNS.items():
        lo, hi = SANE_RANGE[name]
        values = [_to_float(m) for m in re.findall(pattern, text, flags=re.I)]
        values = [v for v in values if v == v and lo <= v <= hi]
        out[name] = max(values) if values else np.nan

    if np.isnan(out["beds"]):
        m = WORD_BED.search(text)
        if m:
            out["beds"] = float(WORD_NUMBERS[m.group(1).lower()])
    if np.isnan(out["baths"]):
        m = WORD_HALF_BATH.search(text) or WORD_BATH.search(text)
        if m:
            half = 0.5 if WORD_HALF_BATH.match(m.group(0)) else 0.0
            out["baths"] = float(WORD_NUMBERS[m.group(1).lower()]) + half

    years = [int(y) for y in YEAR_PATTERN.findall(text)]
    out["year_min"] = float(min(years)) if years else np.nan
    out["year_max"] = float(max(years)) if years else np.nan

    built = [int(g) for m in BUILT_YEAR.findall(text) for g in m if g]
    out["year_built"] = float(min(built)) if built else np.nan
    return out


def fact_frame(texts: pd.Series) -> pd.DataFrame:
    return pd.DataFrame([extract_facts(t) for t in texts], index=texts.index)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def to_log(price: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(price, dtype=float), PRICE_FLOOR, None))


def from_log(log_price: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(log_price, dtype=float))
