"""13 - Duplicates beyond the fold split: how much does uniform tile sampling over-weight the same ground?
Tiles are cut at stride 512 (half a tile) -> every ground 'cell' (512x512) is covered by 1..4 tiles.
Uniform tile sampling + uniform crop position => a cell is seen proportional to its coverage count, so
the 93-tile strip is memorised more than isolated tiles. Proposes per-tile sampling weights that make the
ground density uniform, checks label/appearance of over-weighted cells, and re-verifies fold hygiene."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from common import *

F = pd.read_csv(out("03d_folds.csv")); Y = train_masks(); X = train_images()
cov = defaultdict(list)
for i, r in F.iterrows():
    for dx in (0, 1):
        for dy in (0, 1): cov[(r.comp, r.hx + dx, r.hy + dy)].append(i)
cells = pd.Series({k: len(v) for k, v in cov.items()})
print(f"ground cells (512x512): {len(cells)} | coverage count share: {cells.value_counts(normalize=True).sort_index().round(3).to_dict()}")
print(f"unique ground = {len(cells) / 4:.1f} tile-equivalents (vs 123 tiles, redundancy x{123 / (len(cells) / 4):.2f})")

# sampling density per cell (prob of a random crop pixel landing in that cell) under 2 schemes
def density(w):
    d = defaultdict(float)
    for i, r in F.iterrows():
        for dx in (0, 1):
            for dy in (0, 1): d[(r.comp, r.hx + dx, r.hy + dy)] += w[i] / 4
    s = pd.Series(d); return s / s.sum()
w_uni = np.ones(len(F))
w_bal = np.array([np.mean([1 / len(cov[(r.comp, r.hx + dx, r.hy + dy)]) for dx in (0, 1) for dy in (0, 1)]) for _, r in F.iterrows()])
d_uni, d_bal = density(w_uni), density(w_bal)
ideal = 1 / len(cells)
for n, d in (("uniform tile sampling", d_uni), ("coverage-balanced weights", d_bal)):
    print(f"{n:28s} max/min cell density {d.max() / d.min():.2f} | CV {d.std() / d.mean():.3f} | effective #cells {1 / (d ** 2).sum():.0f} of {len(cells)}")

# is the over-weighted ground different? paddy rate per cell vs its coverage
rows = []
for (c, hx, hy), ids in cov.items():
    i = ids[0]; r = F.loc[i]; oy, ox = (hy - r.hy) * 512, (hx - r.hx) * 512
    rows.append(dict(cov=len(ids), paddy=Y[i][oy:oy + 512, ox:ox + 512].mean(), valid=(X[i][oy:oy + 512, ox:ox + 512].max(-1) > 0).mean()))
C = pd.DataFrame(rows)
print("paddy rate / valid share by coverage count:"); print(C.groupby("cov")[["paddy", "valid"]].mean().round(3).to_string())
print(f"global paddy rate seen by the model: uniform {np.average(C.paddy, weights=d_uni.values):.3f} vs balanced {np.average(C.paddy, weights=d_bal.values):.3f}")

# fold hygiene: train/val share no cell (purge) and how many train cells touch a val cell edge-wise (context leakage)
for k in range(5):
    va = set(F.index[F.fold == k]); tr = set(F.index[(F.fold != k) & ~F[f"purge_{k}"]])
    vcells = {key for key, ids in cov.items() if set(ids) & va}; tcells = {key for key, ids in cov.items() if set(ids) & tr}
    touch = sum(any((c, hx + a, hy + b) in vcells for a, b in ((1, 0), (-1, 0), (0, 1), (0, -1))) for c, hx, hy in tcells - vcells)
    print(f"fold {k}: shared cells train/val = {len(vcells & tcells)} | train cells edge-adjacent to val = {touch}/{len(tcells)}")

F["w_balanced"] = w_bal / w_bal.mean(); F[["id", "w_balanced"]].to_csv(out("13_sampling_weights.csv"), index=False)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].bar(*np.unique(cells, return_counts=True)); ax[0].set_title("tiles covering each 512 ground cell"); ax[0].set_xlabel("coverage")
ax[1].hist([d_uni / ideal, d_bal / ideal], 20, label=["uniform", "balanced"]); ax[1].set_title("cell sampling density / ideal"); ax[1].legend()
ax[2].scatter(F.hy, F.hx + 20 * F.comp, c=w_bal, cmap="viridis", s=30); ax[2].set_title("tile weight (balanced) on layout"); ax[2].set_xlabel("hy"); ax[2].set_ylabel("hx (+20*comp)")
plt.tight_layout(); plt.savefig(out("13_duplicate_sampling.png"), dpi=100); plt.close()
print("weights:", np.round(np.percentile(F.w_balanced, [0, 25, 50, 75, 100]), 2), "saved 13_sampling_weights.csv, 13_duplicate_sampling.png")
