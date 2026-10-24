"""EDA 08 - where the remaining error lives and how low MAE can realistically go.

Two questions the model ladder cannot answer on its own: which listings are
still mispriced after a competent baseline, and how much of that error is
irreducible because the text simply does not contain the price.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRICE_FLOOR, SEED, TABLES, fact_frame, load_raw, repair_text, save_fig, save_table  # noqa: E402

train, _ = load_raw()
train["clean"] = train["text"].map(repair_text)
y = train["listPrice"].to_numpy(dtype=float)
log_y = np.log(np.clip(y, PRICE_FLOOR, None))
fold = pd.read_csv(TABLES / "05_fold_assignment.csv").set_index("id").loc[train["id"], "fold"].to_numpy()
splits = [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(5)]

print("=" * 78)
print("08  ERROR ANATOMY AND THE REALISTIC CEILING")
print("=" * 78)

word = TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                       strip_accents="unicode", max_features=300_000)
char = TfidfVectorizer(sublinear_tf=True, min_df=3, analyzer="char_wb",
                       ngram_range=(3, 5), max_features=300_000)
Xw = word.fit_transform(train["clean"])
Xc = char.fit_transform(train["clean"])

facts = fact_frame(train["clean"])
dense = pd.DataFrame(index=train.index)
for col in facts.columns:
    dense[f"{col}_value"] = facts[col].fillna(facts[col].median())
    dense[f"{col}_missing"] = facts[col].isna().astype(float)
dense["log_sqft"] = np.log1p(facts["sqft"].fillna(0))
dense["log_acres"] = np.log1p(facts["acres"].fillna(0))
dense["char_count"] = train["clean"].str.len()
dense["word_count"] = train["clean"].str.split().str.len()
dense["digit_count"] = train["clean"].str.count(r"\d")
dense["redactions"] = train["clean"].str.count(r"\[Redacted Entity\]")
dense["upper_ratio"] = train["clean"].str.count(r"[A-Z]") / dense["char_count"].clip(lower=1)
Xd = csr_matrix(((dense - dense.mean()) / dense.std().replace(0, 1)).to_numpy())

ladder = {
    "word tfidf": Xw,
    "word + char tfidf": hstack([Xw, Xc]).tocsr(),
    "word + char + extracted facts": hstack([Xw, Xc, Xd]).tocsr(),
}
print("\n--- feature ladder (ridge on log price, shared folds) ---------------")
oof_store = {}
rows = []
for name, matrix in ladder.items():
    oof = np.zeros(len(train))
    for tr_idx, va_idx in splits:
        oof[va_idx] = Ridge(alpha=1.0).fit(matrix[tr_idx], log_y[tr_idx]).predict(matrix[va_idx])
    oof_store[name] = oof
    rows.append({"features": name, "oof_mae": mean_absolute_error(y, np.exp(oof)),
                 "median_ae": float(np.median(np.abs(y - np.exp(oof))))})
    print(f"  {name:32s} MAE = {rows[-1]['oof_mae']:>9,.0f}   medAE = {rows[-1]['median_ae']:>9,.0f}")
save_table(pd.DataFrame(rows), "08_feature_ladder")

best = oof_store["word + char + extracted facts"]
pred = np.exp(best)
err = np.abs(y - pred)

print("\n--- error by true price band ----------------------------------------")
bands = [0, 150e3, 300e3, 500e3, 800e3, 1.5e6, 5e6, np.inf]
rows = []
for lo, hi in zip(bands[:-1], bands[1:]):
    m = (y >= lo) & (y < hi)
    rows.append({
        "band_low": lo, "rows": int(m.sum()),
        "mean_abs_error": float(err[m].mean()),
        "median_ratio_pred_true": float(np.median(pred[m] / y[m])),
        "contribution_to_mae": float(err[m].sum() / len(y)),
        "share_of_mae": float(err[m].sum() / err.sum()),
    })
anatomy = pd.DataFrame(rows)
print(anatomy.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
save_table(anatomy, "08_error_by_band")
print("The model is nearly unbiased in the middle and badly short at the top.")
print("Fixing the top two bands is worth more than every other improvement combined.")

print("\n--- the 25 worst rows -----------------------------------------------")
worst = train.iloc[np.argsort(err)[::-1][:25]].copy()
worst["predicted"] = pred[np.argsort(err)[::-1][:25]]
worst["abs_error"] = err[np.argsort(err)[::-1][:25]]
worst["excerpt"] = worst["clean"].str.slice(0, 110)
print(worst[["id", "listPrice", "predicted", "abs_error", "excerpt"]].to_string(
    index=False, float_format=lambda v: f"{v:,.0f}"))
save_table(worst[["id", "listPrice", "predicted", "abs_error", "text"]], "08_worst_errors")
top25_share = err[np.argsort(err)[::-1][:25]].sum() / err.sum()
print(f"\n25 rows out of {len(train):,} carry {top25_share:.1%} of all absolute error.")

print("\n--- irreducible error: what do twins disagree on? --------------------")
twins = pd.read_csv(TABLES / "05_train_near_duplicate_pairs.csv")
ratio = twins[["left_price", "right_price"]].max(axis=1) / twins[["left_price", "right_price"]].min(axis=1)
gap = (twins["left_price"] - twins["right_price"]).abs()
print(f"near-duplicate pairs              : {len(twins)}")
print(f"median |price difference|         : {gap.median():,.0f}")
print(f"mean   |price difference|         : {gap.mean():,.0f}")
print(f"share of pairs within 1%          : {(ratio <= 1.01).mean():.1%}")
print(f"share of pairs differing by >1.5x : {(ratio > 1.5).mean():.1%}")
print("Two listings whose text is effectively identical still disagree by a")
print(f"mean of ${gap.mean():,.0f}. No model reading only text can do better than")
print("that on those rows; it is a floor, not a target.")
save_table(pd.DataFrame({"abs_gap": gap, "ratio": ratio}), "08_twin_disagreement")

print("\n--- what the error correlates with ----------------------------------")
diag = pd.DataFrame({
    "abs_error": err,
    "log_true": log_y,
    "char_count": dense["char_count"].to_numpy(),
    "n_facts_found": facts.notna().sum(axis=1).to_numpy(),
    "redactions": dense["redactions"].to_numpy(),
})
print(diag.corr(method="spearman")["abs_error"].sort_values(ascending=False).to_string(
    float_format=lambda v: f"{v:+.3f}"))
print("Absolute error tracks the price itself far more than any text property:")
print("the problem is scale, not readability.")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
axes[0].bar([f"{r.band_low/1e3:,.0f}k" for r in anatomy.itertuples()], anatomy["share_of_mae"], color="tab:red")
axes[0].set_title("share of OOF MAE by true price band")
axes[0].tick_params(axis="x", rotation=45)

axes[1].scatter(y, pred / y, s=3, alpha=0.15)
axes[1].axhline(1.0, color="k", ls="--")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("true price")
axes[1].set_ylabel("predicted / true")
axes[1].set_title("systematic under-pricing of expensive homes")

axes[2].hist(np.log10(ratio.clip(lower=1.0)), bins=40, color="tab:purple")
axes[2].set_xlabel("log10 price ratio within a near-duplicate pair")
axes[2].set_title("irreducible disagreement between twins")
fig.tight_layout()
print("\nfigure:", save_fig(fig, "08_error_anatomy"))
