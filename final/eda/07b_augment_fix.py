"""07b - Fix the photometric augmentations found broken in 07.
Finding: every real tile ends with a per-tile 1%/98% stretch, so an aug that skips it creates tiles that never
exist (no saturated px, lifted blacks). Fix = op -> restretch. And restretch on a crop is only realistic if the
gain it induces matches the gains really observed between overlapping tiles -> calibrate the crop size."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from scipy.stats import ks_2samp
from common import *
from augs import *

X, Y, T = train_images(), train_masks(), test_images()
bank = load_gain_bank(); rng = np.random.default_rng(7)
real_gain = np.abs(np.log(bank[:, 0, 1]))  # |log gain| of G channel, both directions

# (a) restretch-induced gain vs crop size
def induced_gain(x):
    y = restretch(x); v = (x.max(-1) > 0) & (x[..., 1] > 5) & (x[..., 1] < 250) & (y[..., 1] > 5) & (y[..., 1] < 250)
    return np.polyfit(x[..., 1][v], y[..., 1][v], 1)[0] if v.sum() > 2000 else np.nan
print("|log gain| real overlaps: p50=%.3f p90=%.3f" % tuple(np.percentile(real_gain, [50, 90])))
cal = {}
for s in (384, 512, 640, 768, 896):
    gs = []
    for _ in range(300):
        i = rng.integers(len(X)); y0, x0 = rng.integers(0, 1024 - s + 1, 2)
        c = X[i][y0:y0 + s, x0:x0 + s]
        if (c.max(-1) > 0).mean() < .3: continue
        gs.append(induced_gain(c))
    gs = np.abs(np.log(np.array(gs)[~np.isnan(gs)]))
    cal[s] = ks_2samp(gs, real_gain).statistic
    print(f"crop {s}: induced |log gain| p50={np.median(gs):.3f} p90={np.percentile(gs, 90):.3f}  KS vs real={cal[s]:.3f}")
best = min(cal, key=cal.get); print("-> crop size whose restretch reproduces the real tile-to-tile gain best:", best)

# (b) fixed photometric ops: op -> restretch(full input)
FIX = {
    "clouds_v1 (07)": lambda x: clouds(x, rng),
    "clouds_v2 soft+restretch": lambda x: restretch(clouds(x, rng, cover=(0.05, 0.3), opacity=(0.3, 0.8))),
    "haze_v1 (07)": lambda x: haze(x, rng),
    "haze_v2 +restretch": lambda x: restretch(haze(x, rng)),
    "blur_v1 (07)": lambda x: blur(x, rng),
    "blur_v2 mild+restretch": lambda x: restretch(blur(x, rng, sigma=(0.3, 0.8))),
    "bank_affine_v1 (07)": lambda x: bank_affine(x, bank, rng),
    "seam_v2 +restretch": lambda x: restretch(seam_affine(x, bank, rng)),
}
def tstats(a):
    v = a.max(-1) > 0
    if v.sum() < 100: return None
    px = a[v].astype(np.float32); hsv = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2HSV)[v].astype(np.float32)
    lap = np.abs(cv2.Laplacian(cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY), cv2.CV_32F))[v]
    return dict(mR=px[:, 0].mean(), mG=px[:, 1].mean(), mB=px[:, 2].mean(), sG=px[:, 1].std(), sat=hsv[:, 1].mean(),
                sharp=lap.mean(), sat255=(px.max(1) == 255).mean(), bright=(px.min(1) > 200).mean())
te = pd.DataFrame([tstats(t) for t in T]); base = pd.DataFrame([tstats(x) for x in X])
res = {"original": {c: ks_2samp(base[c], te[c]).statistic for c in te.columns}}
for an, f in FIX.items():
    tr = pd.DataFrame([s for s in (tstats(f(x.copy())) for x in X) if s])
    res[an] = {c: ks_2samp(tr[c], te[c]).statistic for c in te.columns}
# crop-restretch at calibrated size (crops with <30% valid are skipped, fixing the NaN of 07)
rows = []
for x in X:
    for _ in range(3):
        y0, x0 = rng.integers(0, 1024 - best + 1, 2); c = x[y0:y0 + best, x0:x0 + best]
        if (c.max(-1) > 0).mean() >= .3: rows.append(tstats(restretch(c)))
tr = pd.DataFrame(rows); res[f"crop{best}+restretch"] = {c: ks_2samp(tr[c], te[c]).statistic for c in te.columns}
K = pd.DataFrame(res).T.round(3); K["mean_KS"] = K.mean(axis=1).round(3)
print("\nKS (augmented train, applied to 100% tiles) vs test; lower = closer:\n", K.to_string())
K.to_csv(out("07b_aug_ks_fixed.csv"))

reps = [98, 50, 73, 65]
cols = ["original", "clouds_v1 (07)", "clouds_v2 soft+restretch", "haze_v1 (07)", "haze_v2 +restretch", "blur_v2 mild+restretch", "seam_v2 +restretch", f"crop{best}+restretch"]
fig, ax = plt.subplots(len(reps), len(cols), figsize=(3.3 * len(cols), 3.5 * len(reps)))
for r, i in enumerate(reps):
    for c, an in enumerate(cols):
        if an == "original": im = X[i]
        elif an.startswith("crop"):
            y0 = x0 = (1024 - best) // 2; im = restretch(X[i][y0:y0 + best, x0:x0 + best])
        else: im = FIX[an](X[i].copy())
        ax[r, c].imshow(im[::2, ::2]); ax[r, c].set_title(f"train_{i:03d}\n{an}" if c == 0 else an, fontsize=8); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(out("07b_aug_fixed_grid.png"), dpi=60); plt.close()

# side-by-side realism check: synthetic cloud vs the real cloudy test tile
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(T[3]); ax[0].set_title("REAL clouds: test_003")
ax[1].imshow(FIX["clouds_v1 (07)"](X[76].copy())); ax[1].set_title("synthetic v1 (hard alpha, no restretch)")
ax[2].imshow(FIX["clouds_v2 soft+restretch"](X[76].copy())); ax[2].set_title("synthetic v2 (soft alpha + restretch)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.savefig(out("07b_cloud_realism.png"), dpi=60); plt.close()
