"""03d - Spatial block folds with purging.
Each component is cut into contiguous Y-bands (train area is a narrow N-S strip, so Y-bands are the only
way to get several spatially separated blocks). For fold k the training set drops every tile that overlaps
or touches a validation tile (purge/buffer), mimicking the competition's buffer-tile removal."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

K = 5
lay = pd.read_csv(out("03c_layout.csv")); tr = lay[lay.split == "train"].copy()
Y = train_masks(); tr["frac"] = Y.reshape(len(Y), -1).mean(1)
tr["pos"] = Y.reshape(len(Y), -1).sum(1).astype(np.int64)

# contiguous bands: sort each comp by (hy, hx), cut into chunks of ~len/K*size_share
blocks = []
for c, g in tr.groupby("comp"):
    g = g.sort_values(["hy", "hx"])
    nb = max(1, round(len(g) / (len(tr) / (2 * K))))           # ~2 blocks per fold overall
    for b, part in enumerate(np.array_split(g.index.values, nb)):
        blocks.append((c, b, part))
# greedy assign blocks to folds balancing positive pixels
order = sorted(blocks, key=lambda t: -tr.loc[t[2], "pos"].sum())
fold_pos, fold_n = np.zeros(K), np.zeros(K)
for c, b, part in order:
    k = int(np.argmin(fold_pos / tr.pos.sum() + fold_n / len(tr)))
    tr.loc[part, "fold"] = k; tr.loc[part, "block"] = f"{c}_{b}"
    fold_pos[k] += tr.loc[part, "pos"].sum(); fold_n[k] += len(part)
tr["fold"] = tr.fold.astype(int)

def touches(a, b):
    # tiles are 2x2 half-units; they share pixels iff |dx|<2 and |dy|<2 (edge-adjacent tiles share none)
    return a.comp == b.comp and abs(a.hx - b.hx) < 2 and abs(a.hy - b.hy) < 2

rows = []
for k in range(K):
    va = tr[tr.fold == k]
    purge = [i for i, r in tr[tr.fold != k].iterrows() if any(touches(r, v) for _, v in va.iterrows())]
    tr_k = tr[(tr.fold != k) & (~tr.index.isin(purge))]
    rows.append(dict(fold=k, n_val=len(va), n_train=len(tr_k), n_purged=len(purge),
                     val_frac=va.pos.sum() / (len(va) * 1024 ** 2), blocks=sorted(va.block.unique().tolist())))
    tr[f"purge_{k}"] = tr.index.isin(purge)
rep = pd.DataFrame(rows); print(rep.to_string(index=False))
print("purged tiles per fold are dropped from TRAINING only; they are never used for validation either.")
tr.drop(columns=["split"]).to_csv(out("03d_folds.csv"), index=False)

fig, axs = plt.subplots(1, 2, figsize=(14, 8))
for ax, c in zip(axs, tr.groupby("comp").size().sort_values(ascending=False).index[:2]):
    g = tr[tr.comp == c]
    for r in g.itertuples():
        ax.add_patch(plt.Rectangle((r.hx, -r.hy), 2, -2, fill=True, alpha=.25, color=f"C{r.fold}"))
        ax.text(r.hx + 1, -r.hy - 1, r.id[-3:], ha="center", va="center", fontsize=6)
    ax.set_xlim(g.hx.min() - 1, g.hx.max() + 3); ax.set_ylim(-g.hy.max() - 3, -g.hy.min() + 1); ax.set_aspect("equal")
    ax.set_title(f"comp {c}: colour = fold")
plt.tight_layout(); plt.savefig(out("03d_folds.png"), dpi=100); plt.close()
