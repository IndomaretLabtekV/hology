"""05 - Mask morphology & scale.
(a) connected components / holes / smallest structures of the sawah masks (unique tiles only)
(b) label-resolution ceiling: IoU after mask -> downsample -> upsample at factor f (what the decoder output
    stride or input downscale costs at best), incl. ViT patch-16 token grids at 512/768/1024 input
(c) boundary band: share of pixels within d px of a boundary (where errors concentrate)
(d) polygon character: vertex count of simplified contours -> are labels coarse polygons or pixel-accurate?"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from common import *

X, Y = train_images(), train_masks()
folds = pd.read_csv(out("03d_folds.csv"))
nT = len(Y)

comp_areas, hole_areas, widths, nverts, seglen = [], [], [], [], []
for i in range(nT):
    m = Y[i].astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comp_areas += st[1:, cv2.CC_STAT_AREA].tolist()
    nh, labh, sth, _ = cv2.connectedComponentsWithStats((1 - m).astype(np.uint8), connectivity=4)
    hole_areas += sth[1:, cv2.CC_STAT_AREA].tolist()
    dt = ndi.distance_transform_edt(m)
    sk = dt[(dt > 0) & (ndi.maximum_filter(dt, 3) == dt)]
    widths += (2 * sk).tolist()
    cs, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in cs:
        if cv2.contourArea(c) < 100: continue
        ap = cv2.approxPolyDP(c, 1.5, True); nverts.append((len(c), len(ap), cv2.arcLength(c, True)))
        d = np.diff(ap[:, 0, :], axis=0); seglen += np.hypot(d[:, 0], d[:, 1]).tolist()
comp_areas, hole_areas, widths = map(np.array, (comp_areas, hole_areas, widths))
print("== components (8-conn) ==")
print("count:", len(comp_areas), " area px percentiles [1,5,25,50,75,95]:", np.percentile(comp_areas, [1, 5, 25, 50, 75, 95]).astype(int).tolist())
print("components < 256px: %d (%.4f%% of positive px)" % ((comp_areas < 256).sum(), 100 * comp_areas[comp_areas < 256].sum() / comp_areas.sum()))
print("bg components (holes+bg regions) < 256px: %d, < 1024px: %d" % ((hole_areas < 256).sum(), (hole_areas < 1024).sum()))
print("local width of sawah (2*medial dist) px p5/p25/p50:", np.percentile(widths, [5, 25, 50]).round(1).tolist())
nv = np.array(nverts)
print("contours: mean raw pts=%.0f, after approxPolyDP(1.5px)=%.1f pts, mean segment length=%.1f px" % (nv[:, 0].mean(), nv[:, 1].mean(), np.mean(seglen)))
print("-> long straight segments => labels are hand-digitised polygons (coarse), not pixel-accurate")

# (b) resolution ceiling
print("\n== label resolution ceiling (micro IoU after down->up) ==")
ceil = {}
for f in [2, 4, 8, 16, 32]:
    tp = fp = fn = 0
    for i in range(nT):
        m = Y[i].astype(np.float32)
        d = cv2.resize(m, (1024 // f, 1024 // f), interpolation=cv2.INTER_AREA)
        u = cv2.resize(d, (1024, 1024), interpolation=cv2.INTER_LINEAR) > 0.5
        mb = m > 0.5; tp += (u & mb).sum(); fp += (u & ~mb).sum(); fn += (~u & mb).sum()
    ceil[f] = tp / (tp + fp + fn)
    print(f"  stride {f:2d} (grid {1024 // f}x{1024 // f}): IoU ceiling {ceil[f]:.4f}")
for inp in (512, 768, 1024):
    f = 1024 / (inp / 16); print(f"  ViT/16 tokens at input {inp}: 1 token = {f:.0f}px of original -> ceiling ~{np.interp(f, list(ceil), list(ceil.values())):.4f} (linear head, no high-res decoder)")

# (c) boundary band
band_share = {}
for d in (2, 4, 8, 16):
    s = 0
    for i in range(nT):
        m = Y[i].astype(bool); b = ndi.binary_dilation(m, iterations=d) ^ ndi.binary_erosion(m, iterations=d)
        s += b.mean()
    band_share[d] = s / nT
print("\nshare of pixels within +-d px of a boundary:", {k: round(v, 4) for k, v in band_share.items()})
# if a model is perfect except a +-d band fully wrong (worst case), what IoU?
for d in (2, 4, 8):
    tp = fp = fn = 0
    for i in range(nT):
        m = Y[i].astype(bool)
        er = ndi.binary_erosion(m, iterations=d); di = ndi.binary_dilation(m, iterations=d)
        tp += er.sum(); fn += (m & ~er).sum()
    print(f"  predicting eroded-by-{d}px masks (systematic shrink): IoU={tp / (tp + fn):.4f}")

# (d) spatial prior
heat = Y.mean(0)
print("\npositional prior: sawah rate corner-quadrants",
      [round(heat[:512, :512].mean(), 3), round(heat[:512, 512:].mean(), 3), round(heat[512:, :512].mean(), 3), round(heat[512:, 512:].mean(), 3)])

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
ax[0, 0].hist(np.log10(comp_areas), bins=40, color="tab:green"); ax[0, 0].set_title("log10 area of sawah components (px)")
ax[0, 1].hist(np.log10(hole_areas), bins=40, color="tab:gray"); ax[0, 1].set_title("log10 area of background components")
ax[0, 2].hist(widths, bins=60, range=(0, 300)); ax[0, 2].set_title("local width of sawah (px)")
ax[1, 0].plot(list(ceil), list(ceil.values()), marker="o"); ax[1, 0].set_xscale("log", base=2); ax[1, 0].set_title("IoU ceiling vs output stride"); ax[1, 0].grid()
ax[1, 1].imshow(heat, cmap="viridis"); ax[1, 1].set_title("mean mask (positional prior)")
i = int(np.argmax([((Y[k] > 0).mean() > .3) * ((Y[k] > 0).mean() < .6) for k in range(nT)]))
ax[1, 2].imshow(X[i][300:700, 300:700]); ax[1, 2].contour(Y[i][300:700, 300:700], colors="yellow", linewidths=1); ax[1, 2].set_title(f"{TRAIN_IMG[i].stem} label boundary (400px zoom)")
plt.tight_layout(); plt.savefig(out("05_morphology.png"), dpi=90); plt.close()

# zoom-ins of label boundaries on 6 tiles to judge annotation precision
rng = np.random.default_rng(1); pick = rng.choice(nT, 6, replace=False)
fig, ax = plt.subplots(2, 3, figsize=(18, 12))
for a, i in zip(ax.ravel(), pick):
    m = Y[i]; ys, xs = np.where(ndi.binary_dilation(m, iterations=1) ^ m.astype(bool))
    if len(ys) == 0: continue
    k = rng.integers(len(ys)); cy, cx = np.clip(ys[k], 128, 896), np.clip(xs[k], 128, 896)
    a.imshow(X[i][cy - 128:cy + 128, cx - 128:cx + 128]); a.contour(m[cy - 128:cy + 128, cx - 128:cx + 128], colors="yellow", linewidths=1.2)
    a.set_title(f"{TRAIN_IMG[i].stem} @({cx},{cy}) 256px"); a.axis("off")
plt.tight_layout(); plt.savefig(out("05_boundary_zoom.png"), dpi=70); plt.close()
