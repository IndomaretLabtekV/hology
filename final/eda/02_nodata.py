"""02 - NoData (pure black) analysis: is sawah ever labelled inside nodata? how clean is the border?"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from common import *

X, Y, T = train_images(), train_masks(), test_images()
tot_nod = tot_pos_in_nod = 0
near_black = []
for i in range(len(X)):
    nod = X[i].max(-1) == 0
    tot_nod += nod.sum(); tot_pos_in_nod += (Y[i][nod]).sum()
    # pixels that are almost black but not exactly 0 (anti-aliased / compressed border?)
    near_black.append(((X[i].max(-1) > 0) & (X[i].max(-1) <= 10)).mean())
print("total nodata px:", tot_nod, "| labelled sawah inside nodata:", tot_pos_in_nod,
      f"({tot_pos_in_nod / max(tot_nod, 1):.6f})")
print("near-black (1..10) px fraction train mean: %.5f" % np.mean(near_black))

# nodata is a single connected region? or scattered pixels (dark shadows / water)?
for name, A in [("train", X), ("test", T)]:
    nod_frac, ncomp, small = [], [], 0
    for i in range(len(A)):
        nod = A[i].max(-1) == 0
        lab, n = ndi.label(nod)
        sizes = np.bincount(lab.ravel())[1:]
        small += (sizes < 50).sum(); ncomp.append(n); nod_frac.append(nod.mean())
    nod_frac = np.array(nod_frac)
    print(f"[{name}] nodata frac: mean={nod_frac.mean():.3f}, tiles with >1%={np.sum(nod_frac > .01)}, "
          f">50%={np.sum(nod_frac > .5)}, comps/tile median={np.median(ncomp)}, tiny comps(<50px) total={small}")

# effect on metric: valid-area sawah fraction
valid_frac = np.array([(Y[i][X[i].max(-1) > 0]).mean() for i in range(len(X))])
print("sawah fraction inside VALID area: mean %.3f median %.3f" % (valid_frac.mean(), np.median(valid_frac)))

# Is labelled sawah touching the nodata edge? (labels clipped by AOI = same polygon cut)
touch = []
for i in range(len(X)):
    nod = X[i].max(-1) == 0
    if nod.mean() < 0.01: continue
    ring = ndi.binary_dilation(nod, iterations=3) & ~nod
    touch.append(Y[i][ring].mean())
print("sawah rate on 3px ring just outside nodata: %.3f (vs global %.3f)" % (np.mean(touch), Y.mean()))

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].hist([(X[i].max(-1) == 0).mean() for i in range(len(X))], bins=30, alpha=.6, label="train")
ax[0].hist([(T[i].max(-1) == 0).mean() for i in range(len(T))], bins=30, alpha=.6, label="test")
ax[0].set_title("NoData fraction per tile"); ax[0].legend()
ax[1].hist(valid_frac, bins=30, color="tab:green"); ax[1].set_title("Sawah fraction in valid area (train)")
plt.tight_layout(); plt.savefig(out("02_nodata_hist.png"), dpi=110); plt.close()
