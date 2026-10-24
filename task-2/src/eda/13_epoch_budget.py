"""EDA 13 - why the epoch cap is 5 and not the 6 the audit suggested.

The epoch count does two separate jobs in this training loop. It caps how long
training runs, and it sets the length of the cosine learning-rate schedule.
Those jobs conflict the moment a wall-clock budget can cut the run short, and
this script works out which number is right under that constraint.
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
from common import save_fig, save_table  # noqa: E402

print("=" * 78)
print("13  CHOOSING THE EPOCH CAP")
print("=" * 78)

print("\n--- the epoch cap is also the learning-rate schedule ------------------")
print("total_steps = steps_per_epoch * CFG.epochs, and the cosine schedule is")
print("built over exactly that many steps. Set the cap to 6, get stopped at 4 by")
print("the clock, and the run ends two thirds of the way down the cosine, with")
print("the learning rate still high. The low-rate polish that usually produces")
print("the best checkpoint never happens.")

WARMUP = 0.06


def cosine_lr(progress: float, warmup: float = WARMUP) -> float:
    if progress < warmup:
        return progress / warmup
    adjusted = (progress - warmup) / (1.0 - warmup)
    return 0.5 * (1.0 + np.cos(np.pi * adjusted))


rows = []
for cap in [4, 5, 6, 8]:
    for actually_ran in [3, 4, 5, 6]:
        if actually_ran > cap:
            continue
        progress = actually_ran / cap
        rows.append({
            "epoch_cap": cap,
            "epochs_completed": actually_ran,
            "schedule_progress": progress,
            "lr_fraction_at_stop": cosine_lr(progress),
        })
table = pd.DataFrame(rows)
print("\nlearning rate remaining when training stops, as a fraction of peak")
print(table.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
save_table(table, "13_schedule_truncation")
print("\nA run that finishes its schedule always ends at 0.000. A run cut short")
print("ends wherever the clock left it, and that checkpoint is worse than the")
print("same compute spent under a schedule sized to fit.")

print("\n--- the audit's six-epoch advice was measured on a handicapped run ----")
print("v2 trained with the optimizer bug from section 15, which held the clip")
print("coefficient at 0.684 instead of 0.970.")
retained = 0.684 / 0.970
print(f"  effective learning rate in v2      : {retained:.3f} of requested")
print(f"  speed-up once the bug is fixed     : {1 / retained:.2f}x")
print("An epoch under the fixed optimizer moves roughly as far as 1.4 epochs did")
print("under the broken one. The curve that still looked unconverged at epoch 3")
print("was the curve of a model learning at 70% rate. Five honest epochs cover")
print("more ground than the six the audit inferred from that run.")

print("\n--- how fast was the improvement actually decaying? -------------------")
audit = pd.DataFrame({
    "fold": [0, 1, 2, 3, 4],
    "epoch2": [304_531, 280_394, 296_870, 305_029, 253_320],
    "epoch3": [297_517, 270_450, 278_305, 304_024, 248_277],
})
audit["drop"] = audit["epoch2"] - audit["epoch3"]
audit["drop_pct"] = audit["drop"] / audit["epoch2"]
print(audit.to_string(index=False, float_format=lambda v: f"{v:,.4f}" if v < 1 else f"{v:,.0f}"))
print(f"\nmean drop {audit['drop'].mean():,.0f}, but the spread runs from "
      f"{audit['drop'].min():,.0f} to {audit['drop'].max():,.0f}.")
print("Fold 3 had essentially stopped moving. One delta per fold is not a trend,")
print("and it cannot tell us whether epoch 6 would still be paying. Early")
print("stopping answers that question empirically instead of guessing.")

print("\n--- what the clock allows ---------------------------------------------")
print("The binding constraint is wall-clock, not the cap. With early stopping on")
print("3 stale half-epoch evaluations, the cap is a ceiling the run may never")
print("reach. Sizing it to what actually fits is what keeps the schedule honest.")
budget = []
for minutes_per_epoch in [4, 5, 6, 7, 8]:
    for total_gpu_minutes in [90, 120, 150]:
        per_fold = total_gpu_minutes / 5
        fits = per_fold / minutes_per_epoch
        budget.append({
            "minutes_per_epoch_per_fold": minutes_per_epoch,
            "total_gpu_minutes": total_gpu_minutes,
            "minutes_per_fold": per_fold,
            "epochs_that_fit": fits,
        })
fit = pd.DataFrame(budget)
print(fit.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))
save_table(fit, "13_epoch_budget")
print("\nRead the first fold's log, divide its minutes by the epochs it covered,")
print("then find the row. If the answer is below 5, lower CFG.epochs to match")
print("before the remaining folds run, so the schedule completes.")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
grid = np.linspace(0, 1, 200)
for cap in [5, 6]:
    axes[0].plot(grid * cap, [cosine_lr(g) for g in grid], label=f"cap {cap} epochs")
axes[0].axvline(4, color="tab:red", ls="--", label="clock stops you at epoch 4")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("learning rate, fraction of peak")
axes[0].set_title("a cap you cannot reach leaves the LR high")
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].bar(audit["fold"].astype(str), audit["drop"], color="tab:blue")
axes[1].axhline(audit["drop"].mean(), color="k", ls="--", label=f"mean {audit['drop'].mean():,.0f}")
axes[1].set_xlabel("fold")
axes[1].set_ylabel("MAE gained in epoch 3")
axes[1].set_title("improvement per fold, epoch 2 to 3")
axes[1].legend(fontsize=8)
fig.tight_layout()
print("\nfigure:", save_fig(fig, "13_epoch_budget"))
