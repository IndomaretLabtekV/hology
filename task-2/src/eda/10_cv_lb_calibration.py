"""EDA 10 - map cross-validation MAE onto leaderboard MAE.

Five submissions now have both an out-of-fold score and a public leaderboard
score. That is enough to answer two questions that decide the rest of the run:
does the CV ranking transfer, and what leaderboard number should a given OOF be
expected to produce. Add a row to SUBMISSIONS whenever a new score lands.
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
from common import PREP, ROOT, mae, save_fig, save_table  # noqa: E402

SUBMISSIONS = [
    ("v1 ModernBERT, dollar target", 470_941, 567_900),
    ("nb1 stack", 359_772, 434_473),
    ("xlmr stack", 344_282, 415_954),
    ("v2 sparse + lgbm", 326_358, 412_964),
    ("v2 sparse + lgbm + xlmr", 321_199, 402_234),
    ("v2 full, DeBERTa-v3-large", 277_185, 327_089),
]

print("=" * 78)
print("10  CV TO LEADERBOARD CALIBRATION")
print("=" * 78)

board = pd.DataFrame(SUBMISSIONS, columns=["submission", "oof_mae", "lb_mae"])
board["lb_minus_oof"] = board["lb_mae"] - board["oof_mae"]
board["lb_over_oof"] = board["lb_mae"] / board["oof_mae"]
print(board.to_string(index=False, float_format=lambda v: f"{v:,.4f}" if v < 10 else f"{v:,.0f}"))
save_table(board, "10_cv_lb_pairs")

rho = spearmanr(board["oof_mae"], board["lb_mae"]).statistic
print(f"\nrank agreement between OOF and leaderboard: spearman {rho:.4f}")
print("The ordering transfers without a single inversion. Cross-validation can")
print("be trusted to choose between models; only the absolute level is off.")

slope, intercept = np.polyfit(board["oof_mae"], board["lb_mae"], 1)
fitted = intercept + slope * board["oof_mae"]
r2 = 1 - ((board["lb_mae"] - fitted) ** 2).sum() / ((board["lb_mae"] - board["lb_mae"].mean()) ** 2).sum()
print(f"\nlinear fit   LB = {intercept:,.0f} + {slope:.4f} x OOF      R^2 = {r2:.4f}")
print(f"mean offset  {board['lb_minus_oof'].mean():,.0f}  (sd {board['lb_minus_oof'].std():,.0f})")
print(f"mean ratio   {board['lb_over_oof'].mean():.4f}  (sd {board['lb_over_oof'].std():.4f})")
print("\nEDA 06 predicted this. The test set carries a heavier luxury tail than")
print("train, with a 1.30x ratio at the 0.99 quantile, so every model is graded")
print("on rows more expensive than the ones it was validated against.")

print("\n--- how well did the earlier fit extrapolate? ------------------------")
earlier = board.iloc[:-1]
e_slope, e_intercept = np.polyfit(earlier["oof_mae"], earlier["lb_mae"], 1)
newest = board.iloc[-1]
predicted = e_intercept + e_slope * newest["oof_mae"]
print(f"fit on the first {len(earlier)} points predicted LB {predicted:,.0f} for OOF "
      f"{newest['oof_mae']:,.0f}")
print(f"actual was {newest['lb_mae']:,.0f}, so the extrapolation was {predicted - newest['lb_mae']:,.0f} pessimistic")
print("Extrapolating a fit below the range it was built on is unreliable, and it")
print("was. The refit below includes the new point and should be read as a range,")
print("not a forecast.")

print("\n--- what a future OOF is worth on the leaderboard -------------------")
projection = []
for target in [310_000, 300_000, 290_000, 280_000, 270_000, 260_000]:
    projection.append({
        "oof_mae": target,
        "lb_linear": intercept + slope * target,
        "lb_ratio": target * board["lb_over_oof"].mean(),
        "lb_offset": target + board["lb_minus_oof"].mean(),
    })
proj = pd.DataFrame(projection)
print(proj.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
save_table(proj, "10_lb_projection")
print("The three estimators disagree by about 12,000, which is the honest width")
print("of this projection. Note the newest point has a much smaller gap (+49,904)\nthan the older ones (+71,672 to +96,959): a model that handles the luxury tail\nbetter is penalised less by a test set that is richer in it. Further OOF gains\nmay therefore transfer better than this fit suggests.")

print("\n--- per-fold difficulty, needed to read a single-fold number ---------")
train = pd.read_parquet(PREP / "train_prepared.parquet")
y = train["listPrice"].to_numpy(dtype=float)
fold = train["fold"].to_numpy()
oof = {}
for name in ("ridge", "svr", "lgbm"):
    path = ROOT / "outputs" / "v2" / f"oof_{name}.npy"
    if path.exists():
        oof[name] = np.load(path)
xlmr_path = ROOT / "outputs" / "xlmr_analysis" / "oof_preds" / "xlmr_oof.csv"
if xlmr_path.exists():
    oof["xlmr"] = pd.read_csv(xlmr_path).set_index("id").loc[train["id"], "pred_log"].to_numpy(float)

if oof:
    rows = []
    for name, pred in oof.items():
        row = {"model": name, "overall": mae(y, np.exp(pred))}
        for f in range(5):
            m = fold == f
            row[f"fold{f}"] = mae(y[m], np.exp(pred[m]))
        rows.append(row)
    per_fold = pd.DataFrame(rows)
    print(per_fold.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
    save_table(per_fold, "10_per_fold_mae")

    difficulty = {f: float(np.mean([r[f"fold{f}"] / r["overall"] for r in rows])) for f in range(5)}
    print("\nfold difficulty, as a multiple of the overall score")
    for f, ratio in difficulty.items():
        print(f"  fold {f}: {ratio:.4f}")
    print("Fold 0 is the hardest fold by a clear margin, so a good score there is")
    print("worth more than the same score on fold 4.")

    DEBERTA_FOLD0 = 297_517
    scaled = DEBERTA_FOLD0 / difficulty[0]
    print(f"\nv2 DeBERTa reported {DEBERTA_FOLD0:,.0f} on fold 0.")
    print(f"Rescaled by fold-0 difficulty, that projects to about {scaled:,.0f} OOF")
    print("if the remaining folds behave like the other models' folds.")
    best_single = min(rows, key=lambda r: r["fold0"])
    print(f"On the same fold 0 the best existing single model is {best_single['model']} "
          f"at {best_single['fold0']:,.0f}, so DeBERTa is ahead by "
          f"{best_single['fold0'] - DEBERTA_FOLD0:,.0f}.")
    print("\nTwo caveats before celebrating. The checkpoint was selected on that")
    print("same validation fold, which flatters it by a few thousand, and one fold")
    print("is one fold. Treat it as a strong signal, not a settled number.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(board["oof_mae"], board["lb_mae"], s=70, color="tab:blue")
span = np.linspace(board["oof_mae"].min() * 0.8, board["oof_mae"].max() * 1.05, 50)
axes[0].plot(span, intercept + slope * span, "k--", label=f"LB = {intercept:,.0f} + {slope:.3f}xOOF")
for row in board.itertuples():
    axes[0].annotate(row.submission, (row.oof_mae, row.lb_mae), fontsize=7,
                     textcoords="offset points", xytext=(6, -3))
axes[0].set_xlabel("OOF MAE")
axes[0].set_ylabel("public leaderboard MAE")
axes[0].set_title(f"CV transfers to LB (spearman {rho:.2f}, R2 {r2:.3f})")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

if oof:
    for row in rows:
        axes[1].plot(range(5), [row[f"fold{f}"] for f in range(5)], marker="o", label=row["model"])
    axes[1].axhline(DEBERTA_FOLD0, color="tab:red", ls="--", label="v2 DeBERTa fold 0")
    axes[1].set_xlabel("fold")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("per-fold difficulty")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
fig.tight_layout()
print("\nfigure:", save_fig(fig, "10_cv_lb_calibration"))
