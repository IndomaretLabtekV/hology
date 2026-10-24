"""09b - Do errors concentrate near tile borders (missing context)? micro-IoU / error rate vs distance to tile edge.
If yes: predicting with neighbour context (overlap averaging / stitched windows) is worth it at test time."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

cfg = "d4"
d = np.load(CACHE / "half512.npz"); Y, N = d["Y"].astype(bool), d["N"].astype(bool)
folds = pd.read_csv(out("03d_folds.csv"))
P = {}
for k in range(5):
    f = CACHE / f"oof_{cfg}_f{k}.npy"
    if f.exists():
        for i, p in zip(folds.index[folds.fold == k], np.load(f)): P[i] = p.astype(np.float32)
ids = sorted(P)
yy, xx = np.mgrid[:512, :512]; dist = np.minimum.reduce([yy, xx, 511 - yy, 511 - xx])  # half-res px
bins = [0, 16, 32, 64, 128, 256]
rows = []
for lo, hi in zip(bins[:-1], bins[1:]):
    ring = (dist >= lo) & (dist < hi); tp = fp = fn = 0
    for i in ids:
        pr = (P[i] > .4) & ~N[i]; g = Y[i]
        tp += (pr & g & ring).sum(); fp += (pr & ~g & ring).sum(); fn += (~pr & g & ring).sum()
    rows.append(dict(edge_dist_fullres=f"{2 * lo}-{2 * hi}px", iou=tp / (tp + fp + fn), fn_rate=fn / (tp + fn), fp_rate=fp / (tp + fp), share_px=ring.mean()))
R = pd.DataFrame(rows); print(R.round(4).to_string(index=False))
plt.figure(figsize=(6, 4)); plt.plot(R.edge_dist_fullres, R.iou, marker="o"); plt.title("OOF IoU vs distance to tile edge (d4 proxy)")
plt.ylabel("micro-IoU"); plt.grid(); plt.tight_layout(); plt.savefig(out("09b_edge_effect.png"), dpi=100)
