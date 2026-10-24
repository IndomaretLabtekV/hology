"""EDA 12 - where the 277,185 of remaining error actually sits.

The v2 run produced real out-of-fold predictions. This script cuts them by
segment, by scale, and by direction of error, and asks one question of every
cut: is the error systematic enough that a model could learn it from features
available at inference time, or is it noise we should stop chasing.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PREP, ROOT, mae, save_fig, save_table  # noqa: E402

RUN = ROOT / "outputs" / "holomine_v2_rtx5090"
train = pd.read_parquet(PREP / "train_prepared.parquet")
y = train["listPrice"].to_numpy(dtype=float)
log_y = train["log_price"].to_numpy()
fold = train["fold"].to_numpy()

blend = np.load(RUN / "blend_oof.npy")
deberta = np.load(RUN / "oof_deberta_v3_large.npy")
pred = np.exp(blend)
err = np.abs(y - pred)
ratio = pred / y

print("=" * 78)
print("12  RESIDUAL ANATOMY OF THE v2 RUN")
print("=" * 78)
print(f"blend OOF MAE      {mae(y, pred):,.0f}")
print(f"deberta OOF MAE    {mae(y, np.exp(deberta)):,.0f}")
print(f"median abs error   {np.median(err):,.0f}")
print(f"total error mass   {err.sum():,.0f}")

print("\n--- concentration --------------------------------------------------")
order = np.argsort(err)[::-1]
rows = []
for k in [10, 50, 100, 250, 500, 1000]:
    rows.append({"top_k_rows": k, "row_share": k / len(y),
                 "share_of_total_error": err[order[:k]].sum() / err.sum()})
conc = pd.DataFrame(rows)
print(conc.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
save_table(conc, "12_error_concentration")

print("\n--- segments that a model can recognise at inference ------------------")
text = train["clean"]
SEGMENTS = {
    "NYC apartment/townhouse": r"manhattan|central park|fifth avenue|brooklyn|upper east|upper west|midtown|tribeca|soho|gramercy|doorman|pre-?war",
    "land / ranch": r"\bacres?\b|pasture|grazing|timber|farmland|tillable|\branch\b|\bblm\b",
    "multifamily / portfolio": r"\bunits?\b|multi-?family|portfolio|apartment building|cap rate|rent roll",
    "distressed": r"as[- ]is|fixer|handyman|auction|cash only|\btlc\b|needs work",
    "manufactured": r"mobile home|manufactured|double ?wide|modular",
    "new construction": r"new construction|newly built|never lived|to be built",
}
seg_rows = []
for name, pattern in SEGMENTS.items():
    m = text.str.contains(pattern, case=False, regex=True).to_numpy()
    seg_rows.append({
        "segment": name, "rows": int(m.sum()), "row_share": m.mean(),
        "mae": mae(y[m], pred[m]),
        "share_of_total_error": err[m].sum() / err.sum(),
        "median_pred_over_true": float(np.median(ratio[m])),
        "median_true": float(np.median(y[m])),
    })
m_rest = ~text.str.contains("|".join(SEGMENTS.values()), case=False, regex=True).to_numpy()
seg_rows.append({"segment": "none of the above", "rows": int(m_rest.sum()), "row_share": m_rest.mean(),
                 "mae": mae(y[m_rest], pred[m_rest]),
                 "share_of_total_error": err[m_rest].sum() / err.sum(),
                 "median_pred_over_true": float(np.median(ratio[m_rest])),
                 "median_true": float(np.median(y[m_rest]))})
seg = pd.DataFrame(seg_rows).sort_values("share_of_total_error", ascending=False)
print(seg.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
save_table(seg, "12_segment_errors")

print("\nA segment is only worth a correction if its median ratio is far from 1.")
for row in seg.itertuples():
    bias = abs(row.median_pred_over_true - 1.0)
    verdict = "systematic, correctable" if bias > 0.08 else "centred, nothing to correct"
    print(f"  {row.segment:26s} ratio {row.median_pred_over_true:.3f}  {verdict}")

print("\n--- direction of error, by scale -------------------------------------")
bands = [0, 150e3, 300e3, 500e3, 800e3, 1.5e6, 5e6, np.inf]
rows = []
for lo, hi in zip(bands[:-1], bands[1:]):
    m = (y >= lo) & (y < hi)
    rows.append({
        "band_low": lo, "rows": int(m.sum()),
        "mae": mae(y[m], pred[m]),
        "median_ratio": float(np.median(ratio[m])),
        "share_over_predicted": float(np.mean(ratio[m] > 1)),
        "share_of_total_error": err[m].sum() / err.sum(),
    })
band = pd.DataFrame(rows)
print(band.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
save_table(band, "12_error_by_band")

print("\n--- the same cut, conditioned on the PREDICTION instead ---------------")
print("Binning by true price makes any predictor look biased, because the bin")
print("itself is selected on the noise. Binning by the prediction is the cut a")
print("model can actually act on at inference time.")
pred_decile = pd.qcut(blend, 10, labels=False)
cond = pd.DataFrame({"decile": pred_decile, "pred": pred, "true": y}).groupby("decile").agg(
    n=("true", "size"), median_pred=("pred", "median"), median_true=("true", "median"))
cond["pred_over_true"] = cond["median_pred"] / cond["median_true"]
print(cond.to_string(float_format=lambda v: f"{v:,.3f}"))
save_table(cond.reset_index(), "12_conditional_on_prediction")
worst = (cond["pred_over_true"] - 1.0).abs().max()
print(f"\nlargest deviation from 1.0 across prediction deciles: {worst:.3f}")
if worst < 0.10:
    print("Conditioned on what the model actually says, the predictions are")
    print("already centred. The dramatic pattern in the previous table is the")
    print("shrinkage a conditional median is supposed to have, not a defect.")
    print("There is no calibration left to harvest. Improvement has to come from")
    print("the model knowing more, which means training, not post-processing.")

print("\n--- is the tail still shrunk, and would widening it pay? --------------")
for width in [1.00, 1.05, 1.10, 1.15, 1.20]:
    centre = blend.mean()
    widened = centre + width * (blend - centre)
    print(f"  stretch x{width:.2f}  OOF MAE {mae(y, np.exp(widened)):,.0f}")
print("v2's own calibration search already landed near 1.00, and the audit found")
print("the best global multiplier saves about 14 dollars. Global rescaling is done.")

print("\n--- the two failure modes, side by side ------------------------------")
over = ratio > 2.0
under = ratio < 0.5
print(f"over-predicted by more than 2x  : {over.sum():5d} rows, {err[over].sum() / err.sum():.1%} of error")
print(f"under-predicted by more than 2x : {under.sum():5d} rows, {err[under].sum() / err.sum():.1%} of error")
print("\nworst over-predictions")
idx = np.argsort(np.where(over, err, -1))[::-1][:6]
for i in idx:
    print(f"  true {y[i]:>12,.0f}  pred {pred[i]:>12,.0f}  | {train['clean'].iloc[i][:95]}")
print("\nworst under-predictions")
idx = np.argsort(np.where(under, err, -1))[::-1][:6]
for i in idx:
    print(f"  true {y[i]:>12,.0f}  pred {pred[i]:>12,.0f}  | {train['clean'].iloc[i][:95]}")
print("\nBoth directions are populated. A correction that only lifts expensive-")
print("sounding listings would repair the second group and break the first.")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
axes[0].bar(range(len(seg)), seg["share_of_total_error"], color="tab:red")
axes[0].set_xticks(range(len(seg)))
axes[0].set_xticklabels(seg["segment"], rotation=35, ha="right", fontsize=7)
axes[0].set_title("share of OOF error by segment")

axes[1].scatter(y, ratio, s=3, alpha=0.15)
axes[1].axhline(1.0, color="k", ls="--")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("true price")
axes[1].set_ylabel("predicted / true")
axes[1].set_title("v2 residuals, both directions")

axes[2].plot(np.arange(1, len(y) + 1) / len(y), np.cumsum(np.sort(err)[::-1]) / err.sum())
axes[2].set_xscale("log")
axes[2].set_xlabel("fraction of rows, sorted by error")
axes[2].set_ylabel("cumulative share of error")
axes[2].set_title("error concentration")
axes[2].grid(alpha=0.3)
fig.tight_layout()
print("\nfigure:", save_fig(fig, "12_v2_residuals"))
