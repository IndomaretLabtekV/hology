"""EDA 03 - recovering structured property facts from the prose.

The organisers stripped the tabular columns, but the numbers are still in the
text. This script measures how much of the original table can be parsed back,
whether the repair from EDA 02 changes those numbers, and how each recovered
fact relates to price.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRICE_FLOOR, fact_frame, load_raw, repair_text, save_fig, save_table  # noqa: E402

train, test = load_raw()
y = train["listPrice"].to_numpy(dtype=float)
log_y = np.log(np.clip(y, PRICE_FLOOR, None))

raw_facts = fact_frame(train["text"])
fixed_facts = fact_frame(train["text"].map(repair_text))
test_facts = fact_frame(test["text"].map(repair_text))

print("=" * 78)
print("03  STRUCTURED FACTS RECOVERED FROM TEXT")
print("=" * 78)

cover = pd.DataFrame({
    "train_coverage": fixed_facts.notna().mean(),
    "test_coverage": test_facts.notna().mean(),
    "coverage_before_repair": raw_facts.notna().mean(),
})
cover["coverage_gain_from_repair"] = cover["train_coverage"] - cover["coverage_before_repair"]
print(cover.to_string(float_format=lambda v: f"{v:7.3f}"))
save_table(cover.reset_index().rename(columns={"index": "fact"}), "03_fact_coverage")

print("\n--- did the repair change parsed VALUES, not just coverage? ---------")
value_rows = []
for col in fixed_facts.columns:
    both = raw_facts[col].notna() & fixed_facts[col].notna()
    differ = both & (raw_facts[col] != fixed_facts[col])
    value_rows.append({
        "fact": col,
        "parsed_in_both": int(both.sum()),
        "value_changed": int(differ.sum()),
        "median_before": float(raw_facts.loc[differ, col].median()) if differ.any() else np.nan,
        "median_after": float(fixed_facts.loc[differ, col].median()) if differ.any() else np.nan,
    })
changes = pd.DataFrame(value_rows)
print(changes.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
save_table(changes, "03_repair_value_changes")
print("sqft is the fact the repair rescues: '2, 557 square feet' parses as 557")
print("before the fix and 2557 after, i.e. an order of magnitude on the feature")
print("that separates a studio from a mansion.")

print("\n--- relationship with price ----------------------------------------")
rel_rows = []
for col in fixed_facts.columns:
    present = fixed_facts[col].notna().to_numpy()
    values = fixed_facts.loc[present, col].to_numpy()
    rel_rows.append({
        "fact": col,
        "n_present": int(present.sum()),
        "spearman_vs_log_price": float(spearmanr(values, log_y[present]).statistic),
        "median_price_present": float(np.median(y[present])),
        "median_price_absent": float(np.median(y[~present])),
        "presence_ratio": float(np.median(y[present]) / np.median(y[~present])),
    })
rel = pd.DataFrame(rel_rows).sort_values("spearman_vs_log_price", ascending=False)
print(rel.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
save_table(rel, "03_fact_price_relation")
print("Two separate signals: the VALUE of a fact and the mere MENTION of it.")
print("Both must become features - a missing sqft is itself informative.")

print("\n--- sanity of the parsed values ------------------------------------")
print(fixed_facts.describe(percentiles=[0.01, 0.5, 0.99]).to_string(float_format=lambda v: f"{v:,.2f}"))

print("\n--- implied price per square foot ----------------------------------")
has_sqft = fixed_facts["sqft"].notna().to_numpy()
ppsf = y[has_sqft] / fixed_facts.loc[has_sqft, "sqft"].to_numpy()
print(f"rows with sqft            : {has_sqft.sum():,}")
for q in [0.01, 0.1, 0.5, 0.9, 0.99]:
    print(f"  price/sqft q{q:<5}: {np.quantile(ppsf, q):>10,.0f}")
print("The spread of price-per-sqft is two orders of magnitude, which is exactly")
print("the location effect the text only hints at. Area alone cannot price a home.")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
panels = [("sqft", True), ("acres", True), ("beds", False), ("baths", False), ("year_built", False)]

for ax, (col, logx) in zip(axes.ravel(), panels):
    present = fixed_facts[col].notna()
    values = fixed_facts.loc[present, col]
    prices = y[present.to_numpy()]
    ax.scatter(values, prices, s=4, alpha=0.2)
    if logx:
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"{col} vs price (n={present.sum():,})")
    ax.set_xlabel(col)
    ax.grid(alpha=0.25)

ax = axes.ravel()[5]
ax.hist(np.log10(ppsf), bins=70, color="tab:green")
ax.set_title("log10 price per square foot")
ax.set_xlabel("log10 USD/sqft")
fig.tight_layout()
print("\nfigure:", save_fig(fig, "03_structured_facts"))
