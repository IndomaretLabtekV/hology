"""09 - Error analysis + post-processing on proxy OOF (half-res 512, from 08).
(a) where are the errors: boundary band vs interior, per tile type (nodata-heavy, seam, sawah fraction), FP/FN colour
(b) post-processing evaluated on OOF micro-IoU: threshold, min-area removal, hole filling, closing,
    and averaging predictions of overlapping tiles (usable on test: 11 test-test overlap pairs, no labels needed).
usage: python 09_oof_error_postproc.py [cfg=d4]"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi
from common import *

CFG = sys.argv[1] if len(sys.argv) > 1 else "d4"
d = np.load(CACHE / "half512.npz"); Xh, Yh, Nh = d["X"], d["Y"].astype(bool), d["N"].astype(bool)
folds = pd.read_csv(out("03d_folds.csv"))
P = np.full(Yh.shape, np.nan, np.float32)
for k in range(5):
    f = CACHE / f"oof_{CFG}_f{k}.npy"
    if f.exists(): P[folds.index[folds.fold == k]] = np.load(f).astype(np.float32)
ids = np.where(~np.isnan(P[:, 0, 0]))[0]
print(f"cfg={CFG}: OOF tiles available: {len(ids)}")

def iou(pred, ids_):
    tp = (pred[ids_] & Yh[ids_]).sum(); return tp / ((pred[ids_] | Yh[ids_]).sum() + 1e-9)

base = (P > .5) & ~Nh
print("baseline micro-IoU @0.5 + nodata rule: %.4f" % iou(base, ids))

# (a) error location
fp = base & ~Yh; fn = ~base & Yh
band = np.zeros_like(Yh)
for i in ids: band[i] = ndi.binary_dilation(Yh[i], iterations=4) ^ ndi.binary_erosion(Yh[i], iterations=4)
E = (fp | fn)[ids].sum()
print("share of errors within +-4px (=+-8px full-res) of a label boundary: %.3f (band covers %.3f of px)" % (
    (fp | fn)[ids][band[ids]].sum() / E, band[ids].mean()))
big_fp = big_fn = 0
for i in ids:
    for m, acc in ((fp[i], "fp"), (fn[i], "fn")):
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        s = st[1:, cv2.CC_STAT_AREA]
        if acc == "fp": big_fp += s[s >= 1024].sum()
        else: big_fn += s[s >= 1024].sum()
print("errors in blobs >=1024px (half-res, = whole-field mistakes): FP %.3f, FN %.3f of their totals" % (
    big_fp / fp[ids].sum(), big_fn / fn[ids].sum()))
print("FP px: %d  FN px: %d  (FP/FN ratio %.2f)" % (fp[ids].sum(), fn[ids].sum(), fp[ids].sum() / fn[ids].sum()))

sc = pd.read_csv(out("06_seams_clouds.csv")); sc = sc[sc.split == "train"].reset_index(drop=True)
rows = []
for i in ids:
    rows.append(dict(id=folds.id[i], fold=folds.fold[i], frac=Yh[i].mean(), valid=1 - Nh[i].mean(), seam=sc.n_seams[i] > 0,
                     tp=(base[i] & Yh[i]).sum(), fp=fp[i].sum(), fn=fn[i].sum()))
T = pd.DataFrame(rows); T["iou"] = T.tp / (T.tp + T.fp + T.fn + 1e-9)
T["bin"] = pd.cut(T.frac, [-.01, .15, .35, .6, 1.01])
def grp(g): return pd.Series(dict(n=len(g), micro_iou=g.tp.sum() / (g.tp + g.fp + g.fn).sum(), fp=g.fp.sum(), fn=g.fn.sum()))
print("\nby sawah fraction:\n", T.groupby("bin", observed=True)[["tp", "fp", "fn"]].apply(grp).round(3).to_string())
print("by seam:\n", T.groupby("seam")[["tp", "fp", "fn"]].apply(grp).round(3).to_string())
print("worst tiles:\n", T.sort_values("iou").head(8)[["id", "frac", "valid", "seam", "iou", "fp", "fn"]].round(3).to_string(index=False))

# FP / FN colour: are FPs on green trees, bare soil, roofs?
hsv = np.stack([cv2.cvtColor(Xh[i], cv2.COLOR_RGB2HSV) for i in ids])
def desc(m):
    v = hsv[m[ids]]; return f"H={v[:, 0].mean():.0f} S={v[:, 1].mean():.0f} V={v[:, 2].mean():.0f}"
print("\nmean HSV  TP:", desc(base & Yh), "| FP:", desc(fp), "| FN:", desc(fn), "| TN:", desc(~base & ~Yh & ~Nh))

# (b) post-processing grid
def post(Pm, thr=.5, min_area=0, hole=0, close=0):
    out_ = np.zeros_like(Yh)
    for i in ids:
        m = (Pm[i] > thr) & ~Nh[i]
        if close: m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close, close))).astype(bool) & ~Nh[i]
        if min_area:
            n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8)); keep = st[:, 4] >= min_area; keep[0] = False; m = keep[lab]
        if hole:
            n, lab, st, _ = cv2.connectedComponentsWithStats((~m & ~Nh[i]).astype(np.uint8)); fill = st[:, 4] < hole; fill[0] = False; m = m | fill[lab]
        out_[i] = m
    return out_
res = []
for thr in (.3, .35, .4, .45, .5, .55):
    res.append(("thr", thr, iou(post(P, thr), ids)))
best_thr = max([r for r in res], key=lambda r: r[2])[1]
for a in (64, 256, 1024): res.append(("min_area", a, iou(post(P, best_thr, min_area=a), ids)))
for h in (64, 256, 1024): res.append(("fill_holes", h, iou(post(P, best_thr, hole=h), ids)))
for c in (3, 5, 9): res.append(("close", c, iou(post(P, best_thr, close=c), ids)))
R = pd.DataFrame(res, columns=["op", "param", "iou"]); print("\npost-processing (half-res units; area x4 for full-res):\n", R.round(4).to_string(index=False))
# per-fold threshold stability (is the best threshold the same in each fold? if not, tuning = overfitting)
for k in sorted(folds.fold[ids].unique()):
    ik = ids[folds.fold[ids].values == k]
    print(f"fold {k}: best thr", max((.3, .35, .4, .45, .5, .55), key=lambda t: iou((P > t) & ~Nh, ik)))

# overlap averaging among OOF tiles (same fold -> both predicted by same model)
ov = pd.read_csv(out("03b_overlaps.csv")); ov = ov[(ov.kind == "train-train") & (ov.d4 == 0)]
idx = {n: i for i, n in enumerate(folds.id)}
S_, C_ = np.nan_to_num(P.copy()), np.ones_like(P)
n_used = 0
for r in ov.itertuples():
    i, j = idx[r.a], idx[r.b]
    if i not in ids or j not in ids: continue
    dx, dy = int(r.dx) // 2, int(r.dy) // 2; H = 512
    x0, x1 = max(0, -dx), min(H, H - dx); y0, y1 = max(0, -dy), min(H, H - dy)
    # j at (y,x) == i at (y+dy, x+dx)
    S_[j, y0:y1, x0:x1] += P[i, y0 + dy:y1 + dy, x0 + dx:x1 + dx]; C_[j, y0:y1, x0:x1] += 1
    S_[i, y0 + dy:y1 + dy, x0 + dx:x1 + dx] += P[j, y0:y1, x0:x1]; C_[i, y0 + dy:y1 + dy, x0 + dx:x1 + dx] += 1
    n_used += 1
Pav = S_ / C_
aff = np.unique([idx[a] for a in ov.a] + [idx[b] for b in ov.b]); aff = np.intersect1d(aff, ids)
print(f"\noverlap averaging: {n_used} OOF pairs, {len(aff)} tiles touched")
print("  IoU on touched tiles  before %.4f  after %.4f" % (iou((P > best_thr) & ~Nh, aff), iou((Pav > best_thr) & ~Nh, aff)))
print("  IoU on all OOF tiles  before %.4f  after %.4f" % (iou((P > best_thr) & ~Nh, ids), iou((Pav > best_thr) & ~Nh, ids)))

# figure: worst tiles with error map
w = [idx[t] for t in T.sort_values("iou").id.head(6)]
fig, ax = plt.subplots(2, 6, figsize=(24, 8))
for c, i in enumerate(w):
    e = np.zeros((512, 512, 3), np.uint8); e[base[i] & Yh[i]] = (0, 150, 0); e[fp[i]] = (230, 0, 0); e[fn[i]] = (0, 80, 255)
    ax[0, c].imshow(Xh[i]); ax[0, c].set_title(f"{folds.id[i]} IoU={T.set_index('id').iou[folds.id[i]]:.2f}")
    ax[1, c].imshow(e); ax[1, c].set_title("green TP red FP blue FN")
for a in ax.ravel(): a.axis("off")
plt.tight_layout(); plt.savefig(out(f"09_worst_{CFG}.png"), dpi=55); plt.close()
