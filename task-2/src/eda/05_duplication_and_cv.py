"""EDA 05 - duplicates, near-duplicates, and an honest validation split.

Listing corpora are full of re-posts: the same property described twice with a
few words changed. If those land on both sides of a fold boundary the model is
graded on text it has already memorised and CV stops predicting the leaderboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRICE_FLOOR, SEED, load_raw, repair_text, save_fig, save_table  # noqa: E402

train, test = load_raw()
train["clean"] = train["text"].map(repair_text)
test["clean"] = test["text"].map(repair_text)
y = train["listPrice"].to_numpy(dtype=float)
log_y = np.log(np.clip(y, PRICE_FLOOR, None))

print("=" * 78)
print("05  DUPLICATION AND VALIDATION DESIGN")
print("=" * 78)

norm_train = train["clean"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True)
norm_test = test["clean"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True)
print(f"exact duplicate rows in train        : {int(train['text'].duplicated().sum())}")
print(f"normalised duplicate rows in train   : {int(norm_train.duplicated().sum())}")
print(f"normalised duplicate rows in test    : {int(norm_test.duplicated().sum())}")
print(f"test rows normalised-equal to a train row: {int(norm_test.isin(set(norm_train)).sum())}")

vec = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                      strip_accents="unicode", max_features=300_000)
V = vec.fit_transform(pd.concat([train["clean"], test["clean"]], ignore_index=True))
Vtr, Vte = V[: len(train)], V[len(train):]

THRESHOLD = 0.85
nn = NearestNeighbors(n_neighbors=6, metric="cosine").fit(Vtr)
dist, idx = nn.kneighbors(Vtr)
pairs = [(i, int(j), 1.0 - float(d))
         for i, (drow, jrow) in enumerate(zip(dist, idx))
         for d, j in zip(drow[1:], jrow[1:]) if 1.0 - d >= THRESHOLD]
print(f"\nintra-train pairs with cosine >= {THRESHOLD}: {len(pairs)}")

if pairs:
    rows_i = [p[0] for p in pairs] + [p[1] for p in pairs]
    rows_j = [p[1] for p in pairs] + [p[0] for p in pairs]
    graph = coo_matrix((np.ones(len(rows_i)), (rows_i, rows_j)), shape=(len(train), len(train)))
    n_groups, group = connected_components(graph, directed=False)
else:
    n_groups, group = len(train), np.arange(len(train))
sizes = pd.Series(group).value_counts()
clustered = int((sizes[sizes > 1]).sum())
print(f"connected components                 : {n_groups:,}")
print(f"rows sitting inside a multi-row group: {clustered} ({clustered / len(train):.2%})")
print(f"largest group size                   : {sizes.max()}")

pair_tab = pd.DataFrame(pairs, columns=["left", "right", "similarity"])
if not pair_tab.empty:
    pair_tab["left_price"] = y[pair_tab["left"]]
    pair_tab["right_price"] = y[pair_tab["right"]]
    pair_tab["price_ratio"] = pair_tab[["left_price", "right_price"]].max(axis=1) / pair_tab[["left_price", "right_price"]].min(axis=1)
    print("\nprice agreement inside near-duplicate pairs")
    print(f"  identical price      : {(pair_tab['price_ratio'] == 1).mean():.1%}")
    print(f"  within 10%           : {(pair_tab['price_ratio'] <= 1.1).mean():.1%}")
    print(f"  differ by >2x        : {(pair_tab['price_ratio'] > 2).mean():.1%}")
    print("\nsample pairs")
    print(pair_tab.sort_values("similarity", ascending=False).head(8)[
        ["similarity", "left_price", "right_price", "price_ratio"]
    ].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    save_table(pair_tab, "05_train_near_duplicate_pairs")

print("\n--- test rows that echo a train row ---------------------------------")
nn_cross = NearestNeighbors(n_neighbors=1, metric="cosine").fit(Vtr)
dist_c, idx_c = nn_cross.kneighbors(Vte)
sim_c = 1.0 - dist_c.ravel()
cross = pd.DataFrame({
    "test_id": test["id"].to_numpy(),
    "train_id": train["id"].to_numpy()[idx_c.ravel()],
    "similarity": sim_c,
    "train_price": y[idx_c.ravel()],
}).sort_values("similarity", ascending=False)
for th in [0.99, 0.95, 0.90, 0.85]:
    print(f"  test rows with a train twin >= {th}: {int((sim_c >= th).sum())}")
print(cross.head(8).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
save_table(cross[cross["similarity"] >= 0.85], "05_cross_split_twins")
print("These are free predictions: copy the train price for any test row whose")
print("nearest train neighbour is a textual twin, and let the model handle the rest.")

print("\n--- the cost of ignoring the groups ---------------------------------")
Xv = Vtr
results = []
for name, splitter, groups in [
    ("plain KFold", KFold(5, shuffle=True, random_state=SEED), None),
    ("GroupKFold on duplicate clusters", GroupKFold(n_splits=5), group),
]:
    oof = np.zeros(len(train))
    split_iter = splitter.split(Xv, log_y, groups) if groups is not None else splitter.split(Xv)
    for tr_idx, va_idx in split_iter:
        oof[va_idx] = Ridge(alpha=1.0).fit(Xv[tr_idx], log_y[tr_idx]).predict(Xv[va_idx])
    score = mean_absolute_error(y, np.exp(oof))
    results.append({"scheme": name, "oof_mae": score})
    print(f"  {name:34s} OOF MAE = {score:>9,.0f}")
optimism = results[0]["oof_mae"] - results[1]["oof_mae"]
print(f"  optimism from leaking duplicates  = {optimism:>9,.0f}")
save_table(pd.DataFrame(results), "05_cv_scheme_comparison")

print("\n--- fold-to-fold stability: the whales decide the score --------------")
price_bin = pd.qcut(log_y, 20, labels=False, duplicates="drop")
schemes = {
    "GroupKFold": GroupKFold(n_splits=5).split(Xv, log_y, group),
    "StratifiedGroupKFold on price": StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED).split(Xv, price_bin, group),
}
stability = []
chosen_folds = None
for name, split_iter in schemes.items():
    folds = np.zeros(len(train), dtype=int)
    fold_mae = []
    for f, (tr_idx, va_idx) in enumerate(split_iter):
        folds[va_idx] = f
        pred = np.exp(Ridge(alpha=1.0).fit(Xv[tr_idx], log_y[tr_idx]).predict(Xv[va_idx]))
        fold_mae.append(mean_absolute_error(y[va_idx], pred))
    stability.append({
        "scheme": name,
        "mean_fold_mae": float(np.mean(fold_mae)),
        "std_fold_mae": float(np.std(fold_mae)),
        "max_fold_price_mean": float(pd.Series(y).groupby(folds).mean().max()),
        "min_fold_price_mean": float(pd.Series(y).groupby(folds).mean().min()),
    })
    print(f"  {name:30s} fold MAE {np.mean(fold_mae):>9,.0f} +/- {np.std(fold_mae):>8,.0f}")
    if name.startswith("Stratified"):
        chosen_folds = folds
print("Stratifying on the log-price bin keeps every fold's tail exposure similar,")
print("so a CV delta of a few thousand dollars means the model changed, not the split.")
save_table(pd.DataFrame(stability), "05_fold_stability")

fold_frame = pd.DataFrame({"id": train["id"], "group": group, "fold": chosen_folds})
save_table(fold_frame, "05_fold_assignment")
print("\nfold sizes and price levels (stratified group folds)")
summary = fold_frame.assign(price=y).groupby("fold")["price"].agg(["count", "median", "mean", "max"])
print(summary.to_string(float_format=lambda v: f"{v:,.0f}"))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].hist(sim_c, bins=60, color="tab:orange")
axes[0].axvline(0.85, color="k", ls="--")
axes[0].set_title("test -> nearest train cosine similarity")
axes[0].set_yscale("log")

if not pair_tab.empty:
    axes[1].scatter(pair_tab["similarity"], pair_tab["price_ratio"], s=10, alpha=0.5)
    axes[1].set_yscale("log")
    axes[1].axhline(1.0, color="k", ls="--")
    axes[1].set_xlabel("pair similarity")
    axes[1].set_ylabel("price ratio within pair")
    axes[1].set_title("near-duplicates mostly agree on price")

axes[2].bar([r["scheme"] for r in results], [r["oof_mae"] for r in results], color=["tab:red", "tab:green"])
axes[2].set_ylabel("OOF MAE")
axes[2].set_title("validation scheme")
axes[2].tick_params(axis="x", labelsize=8)
fig.tight_layout()
print("\nfigure:", save_fig(fig, "05_duplication"))
