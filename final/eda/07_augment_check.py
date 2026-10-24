"""07 - Apply every candidate augmentation to representative tiles, save the grids, and check numerically
whether the augmented TRAIN distribution moves toward the TEST distribution (or away from it).
Representatives: high-sawah, low-sawah, NoData-heavy, seam, bluish/hazy tile."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from scipy.stats import ks_2samp
from common import *
from augs import *

X, Y, T = train_images(), train_masks(), test_images()
bank = load_gain_bank(); rng = np.random.default_rng(42)
frac = Y.reshape(len(Y), -1).mean(1); valid = np.array([(x.max(-1) > 0).mean() for x in X])
st = pd.read_csv(out("04_tile_color_train.csv"))
reps = {"high sawah": int(np.argmax(frac * (valid > .95))), "low sawah": int(np.argmin(frac + (valid < .95))),
        "nodata-heavy": int(np.argmin(np.abs(valid - .5))), "seam (train_073)": 73,
        "bluish": int(np.argmax((st.mB - st.mR) * (valid > .9)))}
print("representatives:", {k: f"train_{v:03d}" for k, v in reps.items()})

AUGS = {
    "bank affine": lambda x, m: (bank_affine(x, bank, rng), m),
    "restretch 512 crop": lambda x, m: (cv2.resize(restretch(x[:512, :512]), (1024, 1024), interpolation=cv2.INTER_NEAREST), cv2.resize(m[:512, :512], (1024, 1024), interpolation=cv2.INTER_NEAREST)),
    "seam affine": lambda x, m: (seam_affine(x, bank, rng), m),
    "clouds": lambda x, m: (clouds(x, rng), m),
    "haze": lambda x, m: (haze(x, rng), m),
    "blur": lambda x, m: (blur(x, rng), m),
    "hsv jitter (generic)": lambda x, m: (hsv_jitter(x, rng), m),
    "rotate+scale": lambda x, m: rotate_scale(x, m, rng),
}
names = ["original"] + list(AUGS)
fig, ax = plt.subplots(len(reps), len(names), figsize=(3.2 * len(names), 3.4 * len(reps)))
for r, (rn, i) in enumerate(reps.items()):
    for c, an in enumerate(names):
        im, mk = (X[i], Y[i]) if an == "original" else AUGS[an](X[i].copy(), Y[i].copy())
        ax[r, c].imshow(im[::2, ::2]); ax[r, c].contour(mk[::2, ::2], colors="lime", linewidths=.5)
        ax[r, c].set_title(f"{rn}\n{an}" if c == 0 else an, fontsize=8); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(out("07_aug_grid.png"), dpi=60); plt.close()

# mask/image alignment after geometric ops: label boundary must still sit on the same image edge.
# check: mean |grad| of image along mask boundary before vs after rotation (should be ~equal)
def boundary_grad(im, mk):
    g = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    b = cv2.morphologyEx(mk, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool) & (im.max(-1) > 0)
    b &= cv2.erode((im.max(-1) > 0).astype(np.uint8), np.ones((7, 7))).astype(bool)
    return gm[b].mean() if b.sum() > 100 else np.nan
bg0, bg1, bgs = [], [], []
for i in range(0, len(X), 4):
    bg0.append(boundary_grad(X[i], Y[i])); im, mk = rotate_scale(X[i], Y[i], rng); bg1.append(boundary_grad(im, mk))
    im2, mk2 = rotate_scale(X[i], np.roll(Y[i], 25, 0), rng); bgs.append(boundary_grad(im2, mk2))
print("image gradient on label boundary: original %.2f | after rotate+scale %.2f | deliberately misaligned by 25px %.2f"
      % (np.nanmean(bg0), np.nanmean(bg1), np.nanmean(bgs)))

# distribution check: per-tile stats of augmented train vs test (KS statistic, lower = closer)
def tstats(a):
    v = a.max(-1) > 0; px = a[v].astype(np.float32)
    hsv = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2HSV)[v].astype(np.float32)
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY), cv2.CV_32F))[v]
    return dict(mR=px[:, 0].mean(), mG=px[:, 1].mean(), mB=px[:, 2].mean(), sG=px[:, 1].std(), sat=hsv[:, 1].mean(),
                hue=hsv[:, 0].mean(), sharp=lap.mean(), sat255=(px.max(1) == 255).mean(), bright=(px.min(1) > 200).mean())
te = pd.DataFrame([tstats(t) for t in T])
res = {}
for an in ["original"] + list(AUGS):
    if an == "rotate+scale": continue
    rows = []
    for i in range(len(X)):
        im = X[i] if an == "original" else AUGS[an](X[i].copy(), Y[i].copy())[0]
        rows.append(tstats(im))
    tr = pd.DataFrame(rows)
    res[an] = {c: ks_2samp(tr[c], te[c]).statistic for c in te.columns}
K = pd.DataFrame(res).T.round(3); K["mean_KS"] = K.mean(1).round(3)
print("\nKS distance per-tile stats (augmented train vs test); lower = closer to test:\n", K.to_string())
K.to_csv(out("07_aug_ks.csv"))
