"""06 - Acquisition seams and clouds inside tiles.
Seams: long straight horizontal/vertical discontinuities of the per-row/col colour profile (mosaic of scenes).
Clouds/haze: bright + low-saturation + low-texture pixels. Question: do labels ignore them (label = ground truth
polygon, so sawah under a cloud is still sawah)? If yes the model must 'see through' clouds via context."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from common import *

X, Y, T = train_images(), train_masks(), test_images()

def seams(a):
    v = ~nodata_mask(a)
    f = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 2)
    found = []
    for axis in (0, 1):
        # difference between consecutive rows (axis 0) / cols (axis 1), averaged along the line over valid px
        d = np.abs(np.diff(f, axis=axis)).mean(-1)
        vv = (v[1:] & v[:-1]) if axis == 0 else (v[:, 1:] & v[:, :-1])
        num = (d * vv).sum(1 - axis); den = vv.sum(1 - axis)
        prof = np.where(den > 200, num / np.maximum(den, 1), np.nan)
        base = pd.Series(prof).rolling(41, center=True, min_periods=5).median().values
        z = prof / (base + 1e-3)
        for k in np.where(z > 2.2)[0]:
            if 8 < k < 1015: found.append(("row" if axis == 0 else "col", int(k), float(z[k]), int(den[k])))
    return found

def cloud_mask(a):
    hsv = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2HSV)
    g = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY).astype(np.float32)
    tex = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F)), (15, 15))
    c = (hsv[..., 2] > 170) & (hsv[..., 1] < 60) & (tex < 12)
    c = cv2.morphologyEx(c.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    lab, n = ndi.label(c); sz = np.bincount(lab.ravel()); keep = sz >= 1500; keep[0] = False
    return keep[lab]

rows = []
for split, arr in (("train", X), ("test", T)):
    for i, a in enumerate(arr):
        s = seams(a); cm = cloud_mask(a)
        r = dict(split=split, i=i, n_seams=len(s), seams=s[:4], cloud_frac=cm.mean())
        if split == "train":
            r["sawah_under_cloud"] = Y[i][cm].mean() if cm.sum() else np.nan
            ring = ndi.binary_dilation(cm, iterations=25) & ~cm
            r["sawah_ring"] = Y[i][ring].mean() if ring.sum() else np.nan
        rows.append(r)
D = pd.DataFrame(rows); D.drop(columns="seams").to_csv(out("06_seams_clouds.csv"), index=False)
for sp, g in D.groupby("split"):
    print(f"[{sp}] tiles with >=1 straight seam: {(g.n_seams > 0).mean():.1%} | tiles with cloud >1%: {(g.cloud_frac > .01).mean():.1%} | mean cloud frac {g.cloud_frac.mean():.4f}")
tr = D[D.split == "train"].dropna(subset=["sawah_under_cloud"])
w = tr.cloud_frac
print("train: sawah rate UNDER cloud (area-weighted) %.3f vs ring around cloud %.3f vs global %.3f" % (
    np.average(tr.sawah_under_cloud, weights=w), np.average(tr.sawah_ring, weights=w), Y.mean()))
print("-> if under-cloud ~ ring rate, labels were drawn through clouds: the model must use context, not colour.")

# label continuity across seams (labels ignore seams?)
cross = []
for r in D[(D.split == "train") & (D.n_seams > 0)].itertuples():
    for kind, k, z, n in r.seams:
        m = Y[r.i]; a_ = m[k - 3] if kind == "row" else m[:, k - 3]; b_ = m[k + 3] if kind == "row" else m[:, k + 3]
        cross.append((a_ != b_).mean())
print("train: label change rate across detected seams (+-3px): %.4f (random line ~%.4f)" % (np.mean(cross) if cross else np.nan,
      np.mean([(Y[i][500] != Y[i][506]).mean() for i in range(len(Y))])))

fig, ax = plt.subplots(2, 4, figsize=(24, 12))
sel = D[D.n_seams > 0].sort_values("n_seams", ascending=False).groupby("split").head(2)
sel = pd.concat([sel, D.sort_values("cloud_frac", ascending=False).groupby("split").head(2)])
for a, r in zip(ax.ravel(), sel.itertuples()):
    img = (X if r.split == "train" else T)[r.i].copy()
    cm = cloud_mask(img); img[cm] = (0.5 * img[cm] + [128, 0, 128]).astype(np.uint8)
    a.imshow(img)
    for kind, k, z, n in r.seams:
        (a.axhline if kind == "row" else a.axvline)(k, color="yellow", lw=1)
    if r.split == "train": a.contour(Y[r.i], colors="lime", linewidths=.7)
    a.set_title(f"{r.split}_{r.i:03d} seams={r.n_seams} cloud={r.cloud_frac:.3f}"); a.axis("off")
plt.tight_layout(); plt.savefig(out("06_seams_clouds.png"), dpi=60); plt.close()
