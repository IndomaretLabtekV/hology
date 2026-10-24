"""04 - Radiometry.
(a) Overlap pairs show the SAME ground twice: fit per-channel affine / gamma between the two crops to
    measure the per-tile photometric perturbation (this is what colour augmentation must reproduce).
(b) Is the perturbation uniform inside a tile, or piecewise (seam blocks)?
(c) Train vs test colour distribution on valid pixels (domain shift), haze/cloud/saturation stats."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from common import *

X, Y, T = train_images(), train_masks(), test_images()
A = np.concatenate([X, T]); names = [f.stem for f in TRAIN_IMG] + [f.stem for f in TEST_IMG]
idx = {n: i for i, n in enumerate(names)}
ov = pd.read_csv(out("03b_overlaps.csv")); ov = ov[ov.d4 == 0]

rows, resid_maps = [], []
for r in ov.itertuples():
    i, j = idx[r.a], idx[r.b]; dx, dy = int(r.dx), int(r.dy)
    x0, x1 = max(0, -dx), min(1024, 1024 - dx); y0, y1 = max(0, -dy), min(1024, 1024 - dy)
    a = A[i][y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.float32); b = A[j][y0:y1, x0:x1].astype(np.float32)
    v = (a.max(-1) > 0) & (b.max(-1) > 0) & (a.max(-1) < 255) & (b.max(-1) < 255)
    if v.sum() < 5000: continue
    rec = dict(a=r.a, b=r.b, kind=r.kind, mae=np.abs(a - b)[v].mean())
    pred = np.zeros_like(b)
    for c, ch in enumerate("RGB"):
        g, o = np.polyfit(a[..., c][v], b[..., c][v], 1)
        p = g * a[..., c] + o; pred[..., c] = p
        ss = ((b[..., c][v] - p[v]) ** 2).sum(); st = ((b[..., c][v] - b[..., c][v].mean()) ** 2).sum()
        rec[f"gain_{ch}"], rec[f"off_{ch}"], rec[f"r2_{ch}"] = g, o, 1 - ss / st
    # residual after per-tile affine: if a seam block exists, residual is spatially structured
    res = np.abs(pred - b).mean(-1); res[~v] = np.nan
    rec["resid_mae"] = np.nanmean(res)
    # blockiness: residual of 64x64 cells, std across cells relative to mean
    h, w = res.shape; cells = [np.nanmean(res[y:y + 64, x:x + 64]) for y in range(0, h - 63, 64) for x in range(0, w - 63, 64)]
    rec["resid_cell_cv"] = np.nanstd(cells) / (np.nanmean(cells) + 1e-6)
    rec["luma_ratio"] = b[v].mean() / a[v].mean()
    rows.append(rec); resid_maps.append((rec["resid_cell_cv"], r.a, r.b, res, a, b))
df = pd.DataFrame(rows); df.to_csv(out("04_overlap_photometric.csv"), index=False)
print("pairs analysed:", len(df))
print(df[["mae", "gain_R", "gain_G", "gain_B", "off_R", "off_G", "off_B", "r2_R", "r2_G", "r2_B", "resid_mae", "resid_cell_cv", "luma_ratio"]].describe().round(3).T.to_string())
g = df[["gain_R", "gain_G", "gain_B"]].values
print("\nper-channel gain spread inside a pair (max-min) -> hue/white-balance shift: p50=%.3f p90=%.3f" % tuple(np.percentile(g.max(1) - g.min(1), [50, 90])))
print("affine fit explains most of the difference (median R2 per channel):", df[["r2_R", "r2_G", "r2_B"]].median().round(3).tolist())
print("pairs whose residual is spatially blocky (cell CV>0.5, i.e. seam/cloud inside):", (df.resid_cell_cv > 0.5).sum())

# (c) train vs test valid-pixel colour stats
def stats(arr):
    out_ = []
    for a in arr:
        v = a.max(-1) > 0
        px = a[v].astype(np.float32)
        hsv = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2HSV)[v].astype(np.float32)
        lap = cv2.Laplacian(cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY), cv2.CV_32F)[v]
        out_.append([*px.mean(0), *px.std(0), hsv[:, 1].mean(), hsv[:, 2].mean(), (px.min(1) > 200).mean(), np.abs(lap).mean(), v.mean()])
    return pd.DataFrame(out_, columns=["mR", "mG", "mB", "sR", "sG", "sB", "sat", "val", "bright_frac", "sharp", "valid"])
st_tr, st_te = stats(X), stats(T)
st_tr.to_csv(out("04_tile_color_train.csv"), index=False); st_te.to_csv(out("04_tile_color_test.csv"), index=False)
cmp = pd.DataFrame({"train_mean": st_tr.mean(), "test_mean": st_te.mean(), "train_std": st_tr.std(), "test_std": st_te.std()})
cmp["shift_in_train_std"] = (cmp.test_mean - cmp.train_mean) / cmp.train_std
print("\n== per-tile colour statistics: train vs test ==\n", cmp.round(3).to_string())
from scipy.stats import ks_2samp
print("KS p-values:", {c: round(ks_2samp(st_tr[c], st_te[c]).pvalue, 4) for c in st_tr.columns})

# blue-cast tiles (the hazy/bluish acquisition): B mean > R mean
for nm, s in (("train", st_tr), ("test", st_te)):
    print(f"{nm}: bluish tiles (mB>mR+5): {(s.mB > s.mR + 5).mean():.2%}, bright/cloud-ish px frac p90={s.bright_frac.quantile(.9):.3f}")

# sawah vs background colour separability (train, valid px)
rng = np.random.default_rng(0); samp = []
for i in range(len(X)):
    v = X[i].max(-1) > 0; ys, xs = np.where(v); k = rng.choice(len(ys), min(4000, len(ys)), replace=False)
    hsv = cv2.cvtColor(np.ascontiguousarray(X[i]), cv2.COLOR_RGB2HSV)
    samp.append(np.c_[X[i][ys[k], xs[k]], hsv[ys[k], xs[k]], Y[i][ys[k], xs[k]], np.full(len(k), i)])
S = pd.DataFrame(np.concatenate(samp), columns=["R", "G", "B", "H", "S", "V", "y", "tile"])
print("\nmean colour sawah vs bg:\n", S.groupby("y")[["R", "G", "B", "H", "S", "V"]].mean().round(1).to_string())

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
for k, c in enumerate(["R", "G", "B"]):
    ax[0, 0].hist(df[f"gain_{c}"], bins=30, alpha=.5, label=c, color=c.lower() if c != "G" else "g")
    ax[0, 1].hist(df[f"off_{c}"], bins=30, alpha=.5, label=c, color=c.lower() if c != "G" else "g")
ax[0, 0].set_title("per-channel gain between overlapping tiles"); ax[0, 0].legend()
ax[0, 1].set_title("per-channel offset"); ax[0, 1].legend()
ax[0, 2].hist(df.resid_cell_cv, bins=30); ax[0, 2].set_title("residual blockiness after affine fit")
for k, c in enumerate(["mR", "mG", "mB"]):
    ax[1, k].hist(st_tr[c], bins=30, alpha=.5, density=True, label="train"); ax[1, k].hist(st_te[c], bins=30, alpha=.5, density=True, label="test")
    ax[1, k].set_title(f"tile mean {c[-1]} (valid px)"); ax[1, k].legend()
plt.tight_layout(); plt.savefig(out("04_radiometry.png"), dpi=100); plt.close()

# visual: the 3 blockiest + 3 largest-MAE pairs, show a | b | residual
resid_maps.sort(key=lambda t: -t[0]); pick = resid_maps[:3] + sorted(resid_maps, key=lambda t: -np.nanmean(np.abs(t[4] - t[5])))[:3]
fig, ax = plt.subplots(len(pick), 3, figsize=(15, 4.2 * len(pick)))
for r, (cv, na, nb, res, a, b) in enumerate(pick):
    ax[r, 0].imshow(a.astype(np.uint8)); ax[r, 0].set_title(f"{na} (overlap crop)")
    ax[r, 1].imshow(b.astype(np.uint8)); ax[r, 1].set_title(f"{nb} same ground")
    im = ax[r, 2].imshow(res, vmax=np.nanpercentile(res, 99), cmap="magma"); ax[r, 2].set_title(f"|residual| after per-channel affine, cellCV={cv:.2f}")
    for a_ in ax[r]: a_.axis("off")
plt.tight_layout(); plt.savefig(out("04_overlap_pairs.png"), dpi=55); plt.close()

plt.figure(figsize=(7, 6))
sub = S.sample(20000, random_state=0)
plt.scatter(sub.H, sub.S, c=np.where(sub.y == 1, "tab:green", "tab:gray"), s=1, alpha=.3)
plt.xlabel("Hue (0-179)"); plt.ylabel("Saturation"); plt.title("sawah (green) vs background (gray), pixel HSV")
plt.savefig(out("04_hsv_scatter.png"), dpi=100); plt.close()
