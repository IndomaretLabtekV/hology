"""PREP 03 - a submittable CSV from CPU-only models, in about ten minutes.

The transformer run needs a rented GPU. This script produces a leaderboard entry
in the meantime using the same folds, the same target space, and the same
calibration as the notebook, so its score is directly comparable to the
transformer run that replaces it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, PREP, ROOT, SEED, save_table  # noqa: E402

OUT_DIR = ROOT / "outputs" / "v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_parquet(PREP / "train_prepared.parquet")
test = pd.read_parquet(PREP / "test_prepared.parquet")
sample = pd.read_csv(DATA / "sample_submission.csv")

y = train["listPrice"].to_numpy(dtype=float)
log_y = train["log_price"].to_numpy()
fold = train["fold"].to_numpy()
SPLITS = [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(5)]

FEATURE_COLS = [c for c in train.columns if c not in
                {"id", "text", "clean", "model_text", "listPrice", "fold", "group", "log_price"}]

print("=" * 78)
print("PREP 03  CPU SUBMISSION")
print("=" * 78)


def price_mae(log_pred: np.ndarray, index=None) -> float:
    target = y if index is None else y[index]
    return float(mean_absolute_error(target, np.exp(log_pred)))


started = time.time()
corpus = pd.concat([train["model_text"], test["model_text"]], ignore_index=True)
word = TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                       strip_accents="unicode", max_features=300_000)
char = TfidfVectorizer(sublinear_tf=True, min_df=3, analyzer="char_wb",
                       ngram_range=(3, 5), max_features=300_000)
sparse_all = hstack([word.fit_transform(corpus), char.fit_transform(corpus)]).tocsr()
X, X_test = sparse_all[: len(train)], sparse_all[len(train):]
print(f"sparse matrix {X.shape}  ({time.time() - started:.0f}s)")

oof_log, test_log = {}, {}
for name, factory in [
    ("ridge", lambda: Ridge(alpha=1.0)),
    ("svr", lambda: LinearSVR(C=0.5, epsilon=0.0, max_iter=8000, random_state=SEED)),
]:
    oof = np.zeros(len(train))
    pred = np.zeros(len(test))
    for train_idx, valid_idx in SPLITS:
        model = factory().fit(X[train_idx], log_y[train_idx])
        oof[valid_idx] = model.predict(X[valid_idx])
        pred += model.predict(X_test) / len(SPLITS)
    oof_log[name], test_log[name] = oof, pred
    print(f"{name:8s} OOF MAE {price_mae(oof):>10,.0f}  ({time.time() - started:.0f}s)")

svd = TruncatedSVD(n_components=192, random_state=SEED)
projected = svd.fit_transform(sparse_all)
svd_cols = [f"svd_{i}" for i in range(projected.shape[1])]
dense_train = pd.concat([train[FEATURE_COLS].reset_index(drop=True),
                         pd.DataFrame(projected[: len(train)], columns=svd_cols)], axis=1)
dense_test = pd.concat([test[FEATURE_COLS].reset_index(drop=True),
                        pd.DataFrame(projected[len(train):], columns=svd_cols)], axis=1)
print(f"SVD keeps {svd.explained_variance_ratio_.sum():.1%} of sparse variance")

params = {
    "objective": "l1", "metric": "l1", "learning_rate": 0.03, "num_leaves": 63,
    "min_data_in_leaf": 20, "feature_fraction": 0.5, "bagging_fraction": 0.8,
    "bagging_freq": 1, "lambda_l2": 1.0, "verbosity": -1, "seed": SEED,
}
oof = np.zeros(len(train))
pred = np.zeros(len(test))
best_iters = []
for train_idx, valid_idx in SPLITS:
    booster = lgb.train(
        params,
        lgb.Dataset(dense_train.iloc[train_idx], label=log_y[train_idx]),
        num_boost_round=8000,
        valid_sets=[lgb.Dataset(dense_train.iloc[valid_idx], label=log_y[valid_idx])],
        callbacks=[lgb.early_stopping(200, verbose=False)],
    )
    best_iters.append(booster.best_iteration)
    oof[valid_idx] = booster.predict(dense_train.iloc[valid_idx], num_iteration=booster.best_iteration)
    pred += booster.predict(dense_test, num_iteration=booster.best_iteration) / len(SPLITS)
oof_log["lgbm"], test_log["lgbm"] = oof, pred
print(f"{'lgbm':8s} OOF MAE {price_mae(oof):>10,.0f}  best_iter {best_iters}  ({time.time() - started:.0f}s)")

for name in oof_log:
    np.save(OUT_DIR / f"oof_{name}.npy", oof_log[name])
    np.save(OUT_DIR / f"test_{name}.npy", test_log[name])
np.save(OUT_DIR / "train_ids.npy", train["id"].to_numpy())

names = list(oof_log)
oof_matrix = np.column_stack([oof_log[n] for n in names])
test_matrix = np.column_stack([test_log[n] for n in names])
weights = np.full(len(names), 1.0 / len(names))
for _ in range(60):
    improved = False
    for j in range(len(names)):
        for delta in (0.04, -0.04, 0.01, -0.01):
            trial = weights.copy()
            trial[j] = max(trial[j] + delta, 0.0)
            if trial.sum() <= 0:
                continue
            trial = trial / trial.sum()
            if price_mae(oof_matrix @ trial) < price_mae(oof_matrix @ weights) - 1.0:
                weights, improved = trial, True
    if not improved:
        break
blend_oof = oof_matrix @ weights
blend_test = test_matrix @ weights
print("\nblend weights", {n: round(float(w), 3) for n, w in zip(names, weights)})
print(f"blend OOF MAE {price_mae(blend_oof):,.0f}")

iso_oof = np.zeros(len(train))
for train_idx, valid_idx in SPLITS:
    fold_iso = IsotonicRegression(out_of_bounds="clip").fit(blend_oof[train_idx], log_y[train_idx])
    iso_oof[valid_idx] = fold_iso.predict(blend_oof[valid_idx])
full_iso = IsotonicRegression(out_of_bounds="clip").fit(blend_oof, log_y)
iso_test = full_iso.predict(blend_test)

centre = float(iso_oof.mean())
grid = np.linspace(0.90, 1.50, 61)
best_stretch = min(grid, key=lambda s: price_mae(centre + s * (iso_oof - centre)))
final_oof = centre + best_stretch * (iso_oof - centre)
final_test = centre + best_stretch * (iso_test - centre)
print(f"isotonic OOF MAE {price_mae(iso_oof):,.0f}")
print(f"isotonic + stretch x{best_stretch:.2f} OOF MAE {price_mae(final_oof):,.0f}")

price = np.exp(final_test)
dup = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                      strip_accents="unicode", max_features=300_000)
dup_all = dup.fit_transform(pd.concat([train["clean"], test["clean"]], ignore_index=True))
neigh = NearestNeighbors(n_neighbors=1, metric="cosine").fit(dup_all[: len(train)])
dist, idx = neigh.kneighbors(dup_all[len(train):])
similarity = 1.0 - dist.ravel()
twin = similarity >= 0.97
price[twin] = y[idx.ravel()][twin]
print(f"\ntest rows overridden by an exact textual twin: {int(twin.sum())}")

lower, upper = np.quantile(y, 0.0005), np.quantile(y, 0.9999)
price = np.clip(price, lower, upper)

submission = pd.DataFrame({"id": test["id"], "listPrice": price})
assert submission["id"].tolist() == sample["id"].tolist()
assert submission["listPrice"].notna().all() and (submission["listPrice"] > 0).all()
path = OUT_DIR / "submission_cpu.csv"
submission.to_csv(path, index=False)

comparison = pd.DataFrame({
    "quantile": [0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    "train": [np.quantile(y, q) for q in [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]],
    "submission": [np.quantile(price, q) for q in [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]],
})
print("\nprediction spread vs train")
print(comparison.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
save_table(comparison, "prep03_submission_spread")

manifest = {
    "models": {n: price_mae(oof_log[n]) for n in names},
    "blend_weights": {n: float(w) for n, w in zip(names, weights)},
    "blend_oof_mae": price_mae(blend_oof),
    "calibrated_oof_mae": price_mae(final_oof),
    "stretch": float(best_stretch),
    "twin_overrides": int(twin.sum()),
    "minutes": (time.time() - started) / 60.0,
}
(OUT_DIR / "cpu_run_manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nwrote {path}")
print(json.dumps(manifest, indent=2))
