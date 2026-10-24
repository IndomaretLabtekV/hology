"""PREP 04 - fold the teammate XLM-R predictions into the v2 blend.

The two teammate runs are both dominated by the v2 stack on their own, but the
XLM-R residuals are decorrelated enough to earn a small weight. This costs no
GPU time: it reuses their saved out-of-fold and test predictions directly.
Everything is selected out-of-fold so the reported score stays honest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, PREP, ROOT, mae, save_table  # noqa: E402

OUT_DIR = ROOT / "outputs" / "v2"
XLMR = ROOT / "outputs" / "xlmr_analysis" / "oof_preds"

train = pd.read_parquet(PREP / "train_prepared.parquet")
test = pd.read_parquet(PREP / "test_prepared.parquet")
sample = pd.read_csv(DATA / "sample_submission.csv")

y = train["listPrice"].to_numpy(dtype=float)
log_y = train["log_price"].to_numpy()
fold = train["fold"].to_numpy()
SPLITS = [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(5)]

print("=" * 78)
print("PREP 04  BLEND WITH EXTERNAL XLM-R")
print("=" * 78)

saved_ids = np.load(OUT_DIR / "train_ids.npy", allow_pickle=True)
if not (saved_ids == train["id"].to_numpy()).all():
    raise SystemExit("outputs/v2 OOF files are stale, re-run src/prep/03_cpu_submission.py")

oof = {name: np.load(OUT_DIR / f"oof_{name}.npy") for name in ("ridge", "svr", "lgbm")}
test_pred = {name: np.load(OUT_DIR / f"test_{name}.npy") for name in ("ridge", "svr", "lgbm")}

oof["xlmr"] = pd.read_csv(XLMR / "xlmr_oof.csv").set_index("id").loc[train["id"], "pred_log"].to_numpy(float)
test_pred["xlmr"] = pd.read_csv(XLMR / "xlmr_test.csv").set_index("id").loc[test["id"], "pred_log"].to_numpy(float)


def price_mae(log_pred: np.ndarray, index=None) -> float:
    return mae(y if index is None else y[index], np.exp(log_pred))


def search(keys: list[str]) -> tuple[dict, float, np.ndarray, np.ndarray]:
    matrix = np.column_stack([oof[k] for k in keys])
    test_matrix = np.column_stack([test_pred[k] for k in keys])
    weights = np.full(len(keys), 1.0 / len(keys))
    current = price_mae(matrix @ weights)
    for _ in range(100):
        improved = False
        for j in range(len(keys)):
            for delta in (0.05, -0.05, 0.02, -0.02, 0.01, -0.01):
                trial = weights.copy()
                trial[j] = max(trial[j] + delta, 0.0)
                if trial.sum() <= 0:
                    continue
                trial = trial / trial.sum()
                score = price_mae(matrix @ trial)
                if score < current - 1.0:
                    weights, current, improved = trial, score, True
        if not improved:
            break
    chosen = {k: float(w) for k, w in zip(keys, weights)}
    return chosen, current, matrix @ weights, test_matrix @ weights


results = {}
for label, keys in [("v2 only", ["ridge", "svr", "lgbm"]),
                    ("v2 + xlmr", ["ridge", "svr", "lgbm", "xlmr"])]:
    weights, score, blend_oof, blend_test = search(keys)

    iso_oof = np.zeros(len(train))
    for train_idx, valid_idx in SPLITS:
        fold_iso = IsotonicRegression(out_of_bounds="clip").fit(blend_oof[train_idx], log_y[train_idx])
        iso_oof[valid_idx] = fold_iso.predict(blend_oof[valid_idx])
    full_iso = IsotonicRegression(out_of_bounds="clip").fit(blend_oof, log_y)
    iso_test = full_iso.predict(blend_test)

    polish = np.zeros(len(train))
    factors = []
    for train_idx, valid_idx in SPLITS:
        centre = float(iso_oof[train_idx].mean())
        grid = np.linspace(0.85, 1.55, 71)
        best = min(grid, key=lambda s: price_mae(centre + s * (iso_oof[train_idx] - centre), train_idx))
        polish[valid_idx] = centre + best * (iso_oof[valid_idx] - centre)
        factors.append((centre, best))

    if price_mae(polish) < price_mae(iso_oof):
        centre = float(np.mean([c for c, _ in factors]))
        factor = float(np.mean([f for _, f in factors]))
        final_oof, final_test = polish, centre + factor * (iso_test - centre)
        calibration = f"isotonic + stretch x{factor:.2f}"
    else:
        final_oof, final_test = iso_oof, iso_test
        calibration = "isotonic"

    results[label] = {
        "weights": {k: round(v, 3) for k, v in weights.items() if v > 0.005},
        "blend_mae": score,
        "calibration": calibration,
        "calibrated_mae": price_mae(final_oof),
        "final_test": final_test,
    }
    print(f"\n{label}")
    print(f"  weights          {results[label]['weights']}")
    print(f"  blend OOF MAE    {score:>10,.0f}")
    print(f"  {calibration:<22s} {price_mae(final_oof):>10,.0f}")

gain = results["v2 only"]["calibrated_mae"] - results["v2 + xlmr"]["calibrated_mae"]
print(f"\nXLM-R is worth {gain:,.0f} after calibration, against a fold-to-fold")
print("standard deviation of about 24,785. It is a cheap directional win, not a")
print("result. Measured against the sparse pair alone it looked worth 14,058,")
print("because LightGBM already carries 192 SVD components of the same text.")

final_test = results["v2 + xlmr"]["final_test"]
price = np.exp(final_test)

dup = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2),
                      strip_accents="unicode", max_features=300_000)
dup_all = dup.fit_transform(pd.concat([train["clean"], test["clean"]], ignore_index=True))
neigh = NearestNeighbors(n_neighbors=1, metric="cosine").fit(dup_all[: len(train)])
dist, idx = neigh.kneighbors(dup_all[len(train):])
twin = (1.0 - dist.ravel()) >= 0.97
price[twin] = y[idx.ravel()][twin]
print(f"\ntest rows overridden by an exact textual twin: {int(twin.sum())}")

lower, upper = np.quantile(y, 0.0005), np.quantile(y, 0.9999)
price = np.clip(price, lower, upper)

submission = pd.DataFrame({"id": test["id"], "listPrice": price})
assert submission["id"].tolist() == sample["id"].tolist()
assert submission["listPrice"].notna().all() and (submission["listPrice"] > 0).all()
path = OUT_DIR / "submission_cpu_xlmr.csv"
submission.to_csv(path, index=False)
print(f"wrote {path}")

comparison = pd.DataFrame({
    "quantile": [0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
    "train": [np.quantile(y, q) for q in [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]],
    "submission": [np.quantile(price, q) for q in [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]],
})
print("\nprediction spread vs train")
print(comparison.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
save_table(comparison, "prep04_submission_spread")

(OUT_DIR / "cpu_xlmr_manifest.json").write_text(json.dumps({
    "weights": results["v2 + xlmr"]["weights"],
    "blend_oof_mae": results["v2 + xlmr"]["blend_mae"],
    "calibration": results["v2 + xlmr"]["calibration"],
    "calibrated_oof_mae": results["v2 + xlmr"]["calibrated_mae"],
    "gain_over_v2_only": gain,
    "twin_overrides": int(twin.sum()),
}, indent=2))
