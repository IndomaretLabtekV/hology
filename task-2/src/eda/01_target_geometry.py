"""EDA 01 - target geometry and what MAE actually rewards.

The competition metric is MAE on raw dollars while the target is a heavy
right-tailed price distribution. This script measures how heavy, where the
metric mass lives, and which point estimate MAE is actually asking for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRICE_FLOOR, load_raw, mae, save_fig, save_table  # noqa: E402

train, test = load_raw()
y = train["listPrice"].to_numpy(dtype=float)

print("=" * 78)
print("01  TARGET GEOMETRY")
print("=" * 78)
print(f"train rows            : {len(train):,}")
print(f"test rows             : {len(test):,}")
print(f"mean                  : {y.mean():,.0f}")
print(f"median                : {np.median(y):,.0f}")
print(f"std                   : {y.std():,.0f}")
print(f"skewness              : {pd.Series(y).skew():.2f}")
print(f"kurtosis              : {pd.Series(y).kurt():.2f}")
print(f"log-price skewness    : {pd.Series(np.log(np.clip(y, PRICE_FLOOR, None))).skew():.2f}")
print(f"distinct prices       : {len(np.unique(y)):,}")

quantiles = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
qtab = pd.DataFrame({"quantile": quantiles, "listPrice": [np.quantile(y, q) for q in quantiles]})
print("\nquantiles")
print(qtab.assign(listPrice=qtab["listPrice"].map(lambda v: f"{v:,.0f}")).to_string(index=False))
save_table(qtab, "01_target_quantiles")

print("\n--- point estimates under MAE -------------------------------------")
const_rows = []
for label, value in [
    ("mean", y.mean()),
    ("median", float(np.median(y))),
    ("geometric mean", float(np.exp(np.mean(np.log(np.clip(y, PRICE_FLOOR, None)))))),
]:
    const_rows.append({"estimator": label, "value": value, "mae": mae(y, np.full_like(y, value))})
const = pd.DataFrame(const_rows)
print(const.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
save_table(const, "01_constant_baselines")
print("The median wins by construction: MAE is minimised by the conditional median,")
print("not the conditional mean. Any loss that chases the mean is off-target.")

print("\n--- where the metric mass lives ------------------------------------")
bands = [0, 150e3, 300e3, 500e3, 800e3, 1.5e6, 5e6, np.inf]
med = float(np.median(y))
rows = []
for lo, hi in zip(bands[:-1], bands[1:]):
    m = (y >= lo) & (y < hi)
    rows.append({
        "band_low": lo,
        "band_high": hi,
        "rows": int(m.sum()),
        "row_share": m.mean(),
        "median_price": float(np.median(y[m])),
        "mae_contribution": float(np.abs(y[m] - med).sum() / len(y)),
    })
band = pd.DataFrame(rows)
band["mae_share"] = band["mae_contribution"] / band["mae_contribution"].sum()
print(band.to_string(index=False, float_format=lambda v: f"{v:,.3f}" if v < 1 else f"{v:,.0f}"))
save_table(band, "01_mae_mass_by_band")

top_rows = []
order = np.sort(y)[::-1]
for k in [5, 20, 50, 100, 250, 500]:
    top_rows.append({
        "top_k_rows": k,
        "row_share": k / len(y),
        "mae_contribution": float(np.abs(order[:k] - med).sum() / len(y)),
        "share_of_median_baseline": float(np.abs(order[:k] - med).sum() / np.abs(y - med).sum()),
    })
top = pd.DataFrame(top_rows)
print("\nconcentration of error in the richest listings")
print(top.to_string(index=False, float_format=lambda v: f"{v:,.4f}" if v < 1 else f"{v:,.0f}"))
save_table(top, "01_tail_concentration")

print("\n--- price rounding -------------------------------------------------")
for step in [1_000, 5_000, 10_000, 50_000]:
    share = float(np.mean(y % step == 0))
    print(f"prices divisible by {step:>7,}: {share:6.1%}")
ends = pd.Series(y % 1000).value_counts().head(6)
print("\nmost common last-three-digits")
print(ends.to_string())

degenerate = train[train["listPrice"] < PRICE_FLOOR]
print(f"\nrows below the {PRICE_FLOOR:,.0f} sanity floor: {len(degenerate)}")
print(degenerate[["id", "listPrice"]].to_string(index=False))
save_table(degenerate[["id", "listPrice", "text"]], "01_degenerate_prices")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
axes[0, 0].hist(y, bins=120)
axes[0, 0].set_yscale("log")
axes[0, 0].set_title("listPrice (raw) - log count")
axes[0, 0].set_xlabel("USD")

axes[0, 1].hist(np.log10(np.clip(y, PRICE_FLOOR, None)), bins=80, color="tab:orange")
axes[0, 1].set_title("log10 listPrice - near symmetric")
axes[0, 1].set_xlabel("log10 USD")

axes[1, 0].bar(range(len(band)), band["mae_share"], color="tab:red")
axes[1, 0].set_xticks(range(len(band)))
axes[1, 0].set_xticklabels([f"{r.band_low/1e3:,.0f}k+" for r in band.itertuples()], rotation=45, ha="right")
axes[1, 0].set_title("share of median-baseline MAE by price band")

sorted_err = np.sort(np.abs(y - med))[::-1]
axes[1, 1].plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(sorted_err) / sorted_err.sum())
axes[1, 1].set_xscale("log")
axes[1, 1].set_xlabel("fraction of rows (sorted by error, log scale)")
axes[1, 1].set_ylabel("cumulative share of MAE")
axes[1, 1].set_title("MAE is concentrated in very few rows")
axes[1, 1].grid(alpha=0.3)
fig.tight_layout()
print("\nfigure:", save_fig(fig, "01_target_geometry"))
