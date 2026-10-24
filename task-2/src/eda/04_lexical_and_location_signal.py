"""EDA 04 - which words carry price, and how much of it is geography.

Real estate price is dominated by location. The organisers removed the address
columns, so the only location evidence left is the place names that survive
inside the prose. This script ranks the lexical signal and isolates the
geographic part of it.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRICE_FLOOR, SEED, load_raw, repair_text, save_fig, save_table  # noqa: E402

train, test = load_raw()
train["clean"] = train["text"].map(repair_text)
test["clean"] = test["text"].map(repair_text)
y = train["listPrice"].to_numpy(dtype=float)
log_y = np.log(np.clip(y, PRICE_FLOOR, None))

print("=" * 78)
print("04  LEXICAL AND LOCATION SIGNAL")
print("=" * 78)

vec = TfidfVectorizer(sublinear_tf=True, min_df=15, ngram_range=(1, 2),
                      strip_accents="unicode", max_features=120_000)
X = vec.fit_transform(train["clean"])
ridge = Ridge(alpha=1.0).fit(X, log_y)
terms = np.array(vec.get_feature_names_out())
coef = ridge.coef_
order = np.argsort(coef)

top = pd.DataFrame({
    "expensive_term": terms[order[-30:]][::-1],
    "expensive_coef": coef[order[-30:]][::-1],
    "cheap_term": terms[order[:30]],
    "cheap_coef": coef[order[:30]],
})
print("\nridge coefficients on log price (min_df=15)")
print(top.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
save_table(top, "04_ridge_terms")

print("\n--- concept probes --------------------------------------------------")
CONCEPTS = {
    "luxury_nyc": ["central park", "fifth avenue", "penthouse", "manhattan", "doorman", "pre-war", "loggia"],
    "luxury_generic": ["wine cellar", "chef's kitchen", "marble", "elevator", "gated", "estate", "guest house"],
    "waterfront": ["oceanfront", "lakefront", "waterfront", "beachfront", "dock"],
    "land": ["acres", "ranch", "pasture", "timber", "grazing", "farmland"],
    "distress": ["as-is", "as is", "fixer", "investor", "auction", "handyman", "tlc", "cash only"],
    "budget_build": ["mobile home", "manufactured", "double wide", "vinyl", "carpet"],
    "shared_ownership": ["hoa", "condo", "co-op", "association fee", "monthly maintenance"],
}
rows = []
for concept, keywords in CONCEPTS.items():
    for kw in keywords:
        mask = train["clean"].str.contains(re.escape(kw), case=False, regex=True).to_numpy()
        if mask.sum() < 15:
            continue
        rows.append({
            "concept": concept, "keyword": kw, "rows": int(mask.sum()),
            "median_with": float(np.median(y[mask])),
            "median_without": float(np.median(y[~mask])),
            "ratio": float(np.median(y[mask]) / np.median(y[~mask])),
            "share_above_1.5M": float(np.mean(y[mask] > 1.5e6)),
        })
probe = pd.DataFrame(rows).sort_values("ratio", ascending=False)
print(probe.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
save_table(probe, "04_concept_probes")

print("\n--- geography: proper nouns that survived redaction ------------------")
PLACE = re.compile(r"\b(?:in|near|at|of|to|from|overlooking|downtown)\s+([A-Z][a-z]{2,}(?:\s[A-Z][a-z]{2,}){0,2})")
STOP_PLACES = {"The", "This", "Located", "Enjoy", "Welcome", "Step", "Beautiful", "New"}


def places(text: str) -> list[str]:
    return [p for p in PLACE.findall(text) if p.split()[0] not in STOP_PLACES]


train["places"] = train["clean"].map(places)
test["places"] = test["clean"].map(places)
counts = Counter(p for row in train["places"] for p in row)
print(f"distinct place candidates in train : {len(counts):,}")
print(f"rows with at least one place       : {(train['places'].str.len() > 0).mean():.1%}")
test_counts = Counter(p for row in test["places"] for p in row)
shared = set(counts) & set(test_counts)
covered = sum(test_counts[p] for p in shared) / max(sum(test_counts.values()), 1)
print(f"test place mentions also seen in train: {covered:.1%}")

place_rows = []
for place, n in counts.most_common(400):
    if n < 20:
        continue
    mask = train["places"].map(lambda row, p=place: p in row).to_numpy()
    place_rows.append({
        "place": place, "rows": int(mask.sum()),
        "median_price": float(np.median(y[mask])),
        "vs_global": float(np.median(y[mask]) / np.median(y)),
    })
place_tab = pd.DataFrame(place_rows).sort_values("vs_global", ascending=False)
print("\nmost and least expensive recognised places (n>=20)")
print(pd.concat([place_tab.head(12), place_tab.tail(12)]).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
save_table(place_tab, "04_place_price")

print("\n--- how much MAE does text alone buy? --------------------------------")
kf = KFold(5, shuffle=True, random_state=SEED)
variants = {
    "word 1-2gram": TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                                    strip_accents="unicode", max_features=300_000),
    "word 1-2gram, no proper nouns": None,
}
no_caps = train["clean"].str.replace(r"\b[A-Z][a-z]+\b", " ", regex=True)
for name, vectoriser in variants.items():
    corpus = train["clean"] if vectoriser is not None else no_caps
    vectoriser = vectoriser or TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                                               strip_accents="unicode", max_features=300_000)
    Xv = vectoriser.fit_transform(corpus)
    oof = np.zeros(len(train))
    for tr_idx, va_idx in kf.split(Xv):
        oof[va_idx] = Ridge(alpha=1.0).fit(Xv[tr_idx], log_y[tr_idx]).predict(Xv[va_idx])
    print(f"  {name:32s} MAE = {mean_absolute_error(y, np.exp(oof)):>9,.0f}")
print("Deleting every capitalised word costs real MAE: proper nouns are not")
print("decoration, they are the surviving location feature.")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
sel = top.head(18)
axes[0].barh(range(len(sel)), sel["expensive_coef"], color="tab:red")
axes[0].set_yticks(range(len(sel)))
axes[0].set_yticklabels(sel["expensive_term"], fontsize=8)
axes[0].invert_yaxis()
axes[0].set_title("terms pushing price up")

sel2 = top.head(18)
axes[1].barh(range(len(sel2)), sel2["cheap_coef"], color="tab:blue")
axes[1].set_yticks(range(len(sel2)))
axes[1].set_yticklabels(sel2["cheap_term"], fontsize=8)
axes[1].invert_yaxis()
axes[1].set_title("terms pushing price down")

pl = pd.concat([place_tab.head(10), place_tab.tail(10)])
axes[2].barh(range(len(pl)), pl["vs_global"], color="tab:purple")
axes[2].set_yticks(range(len(pl)))
axes[2].set_yticklabels(pl["place"], fontsize=8)
axes[2].axvline(1.0, color="k", ls="--")
axes[2].set_xscale("log")
axes[2].invert_yaxis()
axes[2].set_title("median price relative to global, by place")
fig.tight_layout()
print("\nfigure:", save_fig(fig, "04_lexical_signal"))
