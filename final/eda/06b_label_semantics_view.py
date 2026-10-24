"""06b - Filled label overlays on tiles where 'paddy-looking' plots are unlabelled -> what does the class really mean?"""
import sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

X, Y = train_images(), train_masks()
ids = [int(s) for s in sys.argv[1:]] or [15, 5, 73, 98, 30, 60]
fig, ax = plt.subplots(2, len(ids), figsize=(7 * len(ids), 14))
for k, i in enumerate(ids):
    ax[0, k].imshow(X[i]); ax[0, k].set_title(f"train_{i:03d}")
    ov = X[i].copy(); m = Y[i].astype(bool)
    ov[m] = (0.55 * ov[m] + [0, 115, 0]).astype(np.uint8); ov[~m & (X[i].max(-1) > 0)] = (0.6 * ov[~m & (X[i].max(-1) > 0)] + [100, 0, 0]).astype(np.uint8)
    ax[1, k].imshow(ov); ax[1, k].set_title(f"green=sawah red=background  frac={m.mean():.2f}")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(out(f"06b_semantics_{'_'.join(map(str, ids))}.png"), dpi=45); plt.close()
print("saved")
