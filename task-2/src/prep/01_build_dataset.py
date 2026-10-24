"""PREP 01 - turn the raw text into the three views the models will consume.

View A  model_text  : repaired, normalised prose for the transformer.
View B  sparse      : word + character TF-IDF for the linear models.
View C  facts       : the numeric table parsed back out of the prose.

Everything is fitted on train+test text jointly, which is allowed because the
test text is given; the target never touches the fitting.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PREP, PRICE_FLOOR, TABLES, fact_frame, load_raw, repair_text, save_table  # noqa: E402

train, test = load_raw()

print("=" * 78)
print("PREP 01  BUILDING THE DATASET")
print("=" * 78)

REDACTION = re.compile(r"\[Redacted Entity\]")
UNGLUE_LOWER_UPPER = re.compile(r"(?<=[a-z]{3})(?=[A-Z][a-z]{2})")


def model_text(raw: str) -> str:
    """Prose the transformer sees.

    Beyond the EDA-02 repairs we (a) keep the redaction marker but shorten it to
    a single token so it costs one embedding slot instead of four, and (b) split
    the 'amenitiesThis' run-ons that the scraper produced by dropping a period
    entirely. Casing, punctuation and stopwords are left alone: a pretrained
    encoder was trained on natural text and every normalisation step past this
    point removes signal it already knows how to use.
    """
    text = repair_text(raw)
    text = REDACTION.sub("[ADDRESS]", text)
    text = UNGLUE_LOWER_UPPER.sub(" ", text)
    return text


for frame in (train, test):
    frame["clean"] = frame["text"].map(repair_text)
    frame["model_text"] = frame["text"].map(model_text)

print(f"train rows {len(train):,}   test rows {len(test):,}")
changed = (train["model_text"] != train["text"]).mean()
print(f"train rows whose text changed at all : {changed:.1%}")
print(f"mean length before / after           : {train['text'].str.len().mean():,.0f} -> {train['model_text'].str.len().mean():,.0f}")


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    facts = fact_frame(frame["clean"])
    out = pd.DataFrame(index=frame.index)
    for col in facts.columns:
        out[f"{col}"] = facts[col]
        out[f"{col}_missing"] = facts[col].isna().astype(np.int8)
    out["log_sqft"] = np.log1p(facts["sqft"])
    out["log_acres"] = np.log1p(facts["acres"])
    out["bed_bath_ratio"] = facts["beds"] / facts["baths"]
    out["sqft_per_bed"] = facts["sqft"] / facts["beds"]
    out["facts_found"] = facts.notna().sum(axis=1)

    text = frame["clean"]
    out["char_count"] = text.str.len()
    out["word_count"] = text.str.split().str.len()
    out["sentence_count"] = text.str.count(r"[.!?]") + 1
    out["mean_word_len"] = out["char_count"] / out["word_count"].clip(lower=1)
    out["digit_count"] = text.str.count(r"\d")
    out["comma_count"] = text.str.count(",")
    out["upper_ratio"] = text.str.count(r"[A-Z]") / out["char_count"].clip(lower=1)
    out["exclaim_count"] = text.str.count("!")
    out["redactions"] = text.str.count(r"\[Redacted Entity\]")
    out["dollar_mentions"] = text.str.count(r"\$")
    out["allcaps_words"] = text.str.count(r"\b[A-Z]{4,}\b")

    for name, pattern in CONCEPT_FLAGS.items():
        out[f"kw_{name}"] = text.str.contains(pattern, case=False, regex=True).astype(np.int8)
    return out


CONCEPT_FLAGS = {
    "nyc": r"manhattan|central park|fifth avenue|brooklyn|upper east|upper west|midtown|tribeca|soho|gramercy",
    "doorman_building": r"doorman|white[- ]glove|concierge|pre-?war|full[- ]service building",
    "luxury": r"penthouse|wine cellar|chef'?s kitchen|marble|elevator|guest house|gated|estate|custom built",
    "waterfront": r"oceanfront|lakefront|waterfront|beachfront|riverfront|private dock|water views?",
    "land": r"\bacres?\b|pasture|grazing|timber|farmland|tillable|ranch\b|blm\b",
    "distress": r"as[- ]is|fixer|handyman|investor|auction|cash only|\btlc\b|needs work|sold as",
    "manufactured": r"mobile home|manufactured|double ?wide|single ?wide|modular",
    "shared_fee": r"\bhoa\b|association fee|monthly maintenance|co-?op\b|common charge",
    "new_build": r"newly (?:built|constructed)|new construction|never lived|brand new",
    "rental_income": r"rental income|cap rate|tenant|noi\b|income[- ]producing|cash flow",
    "views": r"\bviews?\b|panoramic|skyline|vista",
    "renovated": r"renovated|remodeled|updated|refinished|restored",
}

train_feats = build_features(train)
test_feats = build_features(test)
print(f"\nengineered feature columns: {train_feats.shape[1]}")
print(train_feats.describe().T[["count", "mean", "50%", "max"]].to_string(float_format=lambda v: f"{v:,.2f}"))

fold = pd.read_csv(TABLES / "05_fold_assignment.csv").set_index("id")
train["fold"] = fold.loc[train["id"], "fold"].to_numpy()
train["group"] = fold.loc[train["id"], "group"].to_numpy()

train_out = pd.concat([train[["id", "text", "clean", "model_text", "listPrice", "fold", "group"]], train_feats], axis=1)
test_out = pd.concat([test[["id", "text", "clean", "model_text"]], test_feats], axis=1)
train_out["log_price"] = np.log(np.clip(train_out["listPrice"], PRICE_FLOOR, None))

train_out.to_parquet(PREP / "train_prepared.parquet", index=False)
test_out.to_parquet(PREP / "test_prepared.parquet", index=False)
meta = {
    "feature_columns": list(train_feats.columns),
    "n_train": len(train_out),
    "n_test": len(test_out),
    "price_floor": PRICE_FLOOR,
    "repairs": ["split_thousands", "glued_sentence", "collapse_whitespace",
                "redaction_to_[ADDRESS]", "unglue_lower_upper"],
}
(PREP / "prep_meta.json").write_text(json.dumps(meta, indent=2))
print(f"\nwrote {PREP / 'train_prepared.parquet'} and {PREP / 'test_prepared.parquet'}")
save_table(train_feats.describe().T.reset_index(), "prep01_feature_summary")
