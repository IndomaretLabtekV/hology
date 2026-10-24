"""02b - Refined NoData = large black connected regions. Check label inside, border behaviour, rule-based zeroing gain."""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from common import *

X, Y, T = train_images(), train_masks(), test_images()
for thr in [256, 1024, 4096, 16384]:
    nod_px = pos = 0
    for i in range(len(X)):
        nd = nodata_mask(X[i], thr); nod_px += nd.sum(); pos += Y[i][nd].sum()
    print(f"min_area={thr:6d}: nodata px={nod_px:9d}  sawah inside={pos:7d} ({pos / nod_px:.5f})")

# where are the remaining positives inside nodata? -> distance to valid region
dists = []
worst = []
for i in range(len(X)):
    nd = nodata_mask(X[i])
    p = Y[i].astype(bool) & nd
    if p.sum():
        d = ndi.distance_transform_edt(nd)[p]
        dists.append(d); worst.append((p.sum(), i))
d = np.concatenate(dists)
print("distance (px) of sawah-in-nodata to valid edge: p50=%.1f p90=%.1f p99=%.1f max=%.1f" % tuple(np.percentile(d, [50, 90, 99, 100])))
worst.sort(reverse=True); print("tiles w/ most sawah-in-nodata:", [(TRAIN_IMG[i].stem, n) for n, i in worst[:5]])

# Scattered black pixels in valid area: sawah or not?
sc_pos = sc_n = 0
for i in range(len(X)):
    sc = (X[i].max(-1) == 0) & ~nodata_mask(X[i])
    sc_n += sc.sum(); sc_pos += Y[i][sc].sum()
print(f"scattered black px (not nodata): {sc_n}, sawah rate {sc_pos / sc_n:.3f}")

# Upper bound of IoU gain from forcing 0 in nodata: simulate a model that leaks p=0.5 into nodata
print("frac of pixels that are nodata - train %.3f, test %.3f" % (
    np.mean([nodata_mask(x).mean() for x in X]), np.mean([nodata_mask(x).mean() for x in T])))

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
n, i = worst[0]
nd = nodata_mask(X[i]); sc = (X[i].max(-1) == 0) & ~nd
vis = X[i].copy(); vis[nd] = [60, 60, 160]; vis[sc] = [255, 0, 0]
ax[0].imshow(vis); ax[0].set_title(f"{TRAIN_IMG[i].stem}: blue=NoData, red=scattered black")
ax[1].imshow(Y[i], cmap="gray"); ax[1].contour(nd, colors="b", linewidths=.5); ax[1].set_title("mask + NoData edge")
ys, xs = np.where(Y[i].astype(bool) & nd); cy, cx = int(np.median(ys)), int(np.median(xs))
sl = np.s_[max(cy - 100, 0):cy + 100, max(cx - 100, 0):cx + 100]
ax[2].imshow(X[i][sl]); ax[2].contour(Y[i][sl], colors="lime", linewidths=1); ax[2].set_title("zoom: mask over nodata")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(out("02b_nodata_refined.png"), dpi=90); plt.close()
