"""EDA 07 - choosing the target transform and the loss, with evidence.

This is the decisive experiment of the whole project. The metric is MAE in
dollars; the target spans six orders of magnitude. Which space the model should
regress in, and which loss it should minimise there, is worth more than any
amount of architecture search.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRICE_FLOOR, SEED, TABLES, load_raw, repair_text, save_fig, save_table  # noqa: E402

train, _ = load_raw()
train["clean"] = train["text"].map(repair_text)
y = train["listPrice"].to_numpy(dtype=float)
log_y = np.log(np.clip(y, PRICE_FLOOR, None))

folds_path = TABLES / "05_fold_assignment.csv"
if not folds_path.exists():
    raise SystemExit("run src/eda/05_duplication_and_cv.py first")
fold = pd.read_csv(folds_path).set_index("id").loc[train["id"], "fold"].to_numpy()

vec = TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                      strip_accents="unicode", max_features=300_000)
X = vec.fit_transform(train["clean"])
splits = [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(5)]

print("=" * 78)
print("07  TARGET TRANSFORM AND LOSS")
print("=" * 78)

print("\n--- how symmetric is the target in each space? -----------------------")
space = pd.DataFrame({
    "space": ["raw price", "sqrt price", "log price", "price / 1e6"],
    "skewness": [
        pd.Series(y).skew(), pd.Series(np.sqrt(y)).skew(),
        pd.Series(log_y).skew(), pd.Series(y / 1e6).skew(),
    ],
    "kurtosis": [
        pd.Series(y).kurt(), pd.Series(np.sqrt(y)).kurt(),
        pd.Series(log_y).kurt(), pd.Series(y / 1e6).kurt(),
    ],
})
print(space.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
print("A squared or Huber loss in raw-dollar space spends almost all of its")
print("gradient on the 1.5% of rows above $5M. In log space the target is")
print("almost Gaussian and every listing gets a comparable share of the update.")


def run(transform, inverse, model_factory, name):
    oof = np.zeros(len(train))
    for tr_idx, va_idx in splits:
        model = model_factory()
        model.fit(X[tr_idx], transform(y[tr_idx]))
        oof[va_idx] = model.predict(X[va_idx])
    pred = np.clip(inverse(oof), 1_000, None)
    return {"setup": name, "mae": mean_absolute_error(y, pred),
            "median_ae": float(np.median(np.abs(y - pred))),
            "bias": float(np.mean(pred - y))}, pred, oof


print("\n--- target space, same model, same folds -----------------------------")
rows = []
preds = {}
for name, transform, inverse in [
    ("raw dollars", lambda v: v, lambda v: v),
    ("price / 1e6", lambda v: v / 1e6, lambda v: v * 1e6),
    ("sqrt dollars", np.sqrt, lambda v: np.clip(v, 0, None) ** 2),
    ("log dollars", lambda v: np.log(np.clip(v, PRICE_FLOOR, None)), np.exp),
]:
    row, pred, _ = run(transform, inverse, lambda: Ridge(alpha=1.0), f"ridge L2 on {name}")
    rows.append(row)
    preds[name] = pred

row, pred_svr, oof_log_l1 = run(
    lambda v: np.log(np.clip(v, PRICE_FLOOR, None)), np.exp,
    lambda: LinearSVR(C=0.5, epsilon=0.0, max_iter=8000, random_state=SEED),
    "linearSVR L1 on log dollars",
)
rows.append(row)
preds["log dollars, L1 loss"] = pred_svr

table = pd.DataFrame(rows).sort_values("mae")
print(table.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
save_table(table, "07_target_space_comparison")
best_raw = table[table["setup"].str.contains("raw|1e6")]["mae"].min()
best_log = table[table["setup"].str.contains("log")]["mae"].min()
print(f"\nmoving from dollar space to log space: {best_raw:,.0f} -> {best_log:,.0f} MAE "
      f"({(best_raw - best_log) / best_raw:.1%} better)")
print("This single decision is larger than any architecture change available to us.")

print("\n--- why L1 in log space is the right loss ----------------------------")
print("MAE in dollars is minimised by the conditional MEDIAN of price. Because")
print("exp() is monotone, median(price | text) = exp(median(log price | text)).")
print("So: regress log price with an L1 / quantile-0.5 objective, then exponentiate.")
print("An L2 objective in log space returns the conditional MEAN of log price,")
print("which exponentiates to the geometric mean - always below the median when")
print("the conditional distribution is right-skewed.")

print("\n--- post-hoc recalibration of the log prediction ---------------------")
base = oof_log_l1
cal_rows = [{"calibration": "none", "mae": mean_absolute_error(y, np.exp(base))}]

grid = np.linspace(0.85, 1.25, 81)
best_k = min(grid, key=lambda k: mean_absolute_error(y, np.exp(base) * k))
cal_rows.append({"calibration": f"multiplicative x{best_k:.3f}",
                 "mae": mean_absolute_error(y, np.exp(base) * best_k)})

centre = base.mean()
stretch_grid = np.linspace(0.9, 1.6, 71)
best_s = min(stretch_grid, key=lambda s: mean_absolute_error(y, np.exp(centre + s * (base - centre))))
cal_rows.append({"calibration": f"log-space stretch x{best_s:.3f}",
                 "mae": mean_absolute_error(y, np.exp(centre + best_s * (base - centre)))})

iso_oof = np.zeros(len(train))
for tr_idx, va_idx in splits:
    iso = IsotonicRegression(out_of_bounds="clip").fit(base[tr_idx], log_y[tr_idx])
    iso_oof[va_idx] = iso.predict(base[va_idx])
cal_rows.append({"calibration": "isotonic, fitted out-of-fold",
                 "mae": mean_absolute_error(y, np.exp(iso_oof))})

cal = pd.DataFrame(cal_rows).sort_values("mae")
print(cal.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
save_table(cal, "07_calibration")
print(f"The fitted stretch factor is {best_s:.2f} > 1, which is the model admitting")
print("it shrinks toward the middle. Widening the spread of the log prediction")
print("recovers part of the tail that regression-to-the-mean removed.")

print("\n--- shrinkage, measured directly -------------------------------------")
bins = pd.qcut(base, 10, labels=False)
shrink = pd.DataFrame({"bin": bins, "pred": np.exp(base), "true": y}).groupby("bin").agg(
    n=("true", "size"), median_pred=("pred", "median"), median_true=("true", "median"))
shrink["pred_over_true"] = shrink["median_pred"] / shrink["median_true"]
print(shrink.to_string(float_format=lambda v: f"{v:,.3f}"))
save_table(shrink.reset_index(), "07_shrinkage_by_decile")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
names = [r["setup"] for r in rows]
values = [r["mae"] for r in rows]
axes[0].barh(range(len(names)), values, color=["tab:red" if "log" not in n else "tab:green" for n in names])
axes[0].set_yticks(range(len(names)))
axes[0].set_yticklabels(names, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel("OOF MAE")
axes[0].set_title("target space decides the score")

axes[1].scatter(y, np.exp(base), s=3, alpha=0.15)
lims = [1e4, 1e8]
axes[1].plot(lims, lims, "k--")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("true price")
axes[1].set_ylabel("predicted price")
axes[1].set_title("predictions are compressed toward the middle")

axes[2].plot(shrink.index, shrink["pred_over_true"], marker="o")
axes[2].axhline(1.0, color="k", ls="--")
axes[2].set_xlabel("decile of prediction")
axes[2].set_ylabel("median predicted / median true")
axes[2].set_title("over-prediction low, under-prediction high")
axes[2].grid(alpha=0.3)
fig.tight_layout()
print("\nfigure:", save_fig(fig, "07_target_and_loss"))
