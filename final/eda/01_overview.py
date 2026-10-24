"""01 - Basic overview: mask integrity, sawah fraction per tile, empty/full tiles, sample grid."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

X, Y, T = train_images(), train_masks(), test_images()
raw = np.load(CACHE / "train_msk.npy", mmap_mode="r")
print("unique raw mask values:", np.unique(raw[::7, ::4, ::4]))

frac = Y.reshape(len(Y), -1).mean(1)
df = pd.DataFrame({"id": [f.stem for f in TRAIN_IMG], "sawah_frac": frac})
df.to_csv(out("01_tile_stats.csv"), index=False)
print("\n== sawah fraction per tile ==")
print(df.sawah_frac.describe().round(4))
print("global pixel positive rate:", round(frac.mean(), 4))
print("empty tiles (<0.1%):", (frac < 0.001).sum(), "| full tiles (>99%):", (frac > 0.99).sum())
print("tiles bins:", np.histogram(frac, bins=[0, .001, .1, .3, .5, .7, .9, .99, 1.0001])[0])

# black / nodata pixels (padding at block borders?)
nod_tr = (X.reshape(len(X), -1, 3).max(-1) == 0).mean(1)
nod_te = (T.reshape(len(T), -1, 3).max(-1) == 0).mean(1)
print("\n== pure-black (nodata) pixel fraction ==")
print("train tiles with nodata:", (nod_tr > 0.001).sum(), "max", nod_tr.max().round(3))
print("test  tiles with nodata:", (nod_te > 0.001).sum(), "max", nod_te.max().round(3))
white_tr = (X.reshape(len(X), -1, 3).min(-1) == 255).mean(1)
print("train tiles with saturated white >1%:", (white_tr > 0.01).sum())

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(frac, bins=30, color="tab:green"); ax[0].set_title("Sawah fraction per train tile"); ax[0].set_xlabel("fraction")
ax[1].plot(np.sort(frac)[::-1], marker="."); ax[1].set_title("Sorted sawah fraction (train)"); ax[1].set_xlabel("rank")
plt.tight_layout(); plt.savefig(out("01_sawah_fraction.png"), dpi=110); plt.close()

# sample grid: image + mask overlay across fraction quantiles
idx = np.argsort(frac)[np.linspace(0, len(frac) - 1, 8).astype(int)]
fig, ax = plt.subplots(2, 8, figsize=(32, 8))
for j, i in enumerate(idx):
    ax[0, j].imshow(X[i]); ax[0, j].set_title(f"{df.id[i]} f={frac[i]:.2f}")
    ov = X[i].copy(); ov[Y[i] == 1] = (0.5 * ov[Y[i] == 1] + [0, 127, 0]).astype(np.uint8)
    ax[1, j].imshow(ov)
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(out("01_samples_by_fraction.png"), dpi=70); plt.close()

# test sample grid
fig, ax = plt.subplots(2, 8, figsize=(32, 8))
for a, i in zip(ax.ravel(), np.linspace(0, len(T) - 1, 16).astype(int)):
    a.imshow(T[i]); a.set_title(TEST_IMG[i].stem); a.axis("off")
plt.tight_layout(); plt.savefig(out("01_test_samples.png"), dpi=70); plt.close()
print("saved figures to", OUT)
