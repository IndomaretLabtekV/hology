"""14 - Augmentation v3, derived from 11/12, applied + inspected + measured.
(a) Proof that global haze + per-channel restretch is a no-op (haze = per-channel affine, restretch inverts it).
(b) Real haze is REGIONAL (one acquisition block of a mosaic tile, tile-level stretch computed over both):
    measure the hazy-block signature on train seam tiles, build haze_block(), compare to real hazy test blocks.
(c) Visual grid: representative tile per type x each v3 op, real hazy test tiles shown next to synthetic ones.
(d) Coverage check: patch-level adversarial AUC test vs train augmented with v2 pipeline vs v3 pipeline
    (lower AUC = augmented train covers the test appearance better). Uses test IMAGES only, never labels."""
import numpy as np, pandas as pd, matplotlib, cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from common import *
from augs import load_gain_bank, restretch, valid_of, clouds

X, Y, T = train_images(), train_masks(), test_images()
rng = np.random.default_rng(0)


def contrast_aug(img, rng, region=None):
    v = valid_of(img) if region is None else valid_of(img) & region
    if v.sum() < 100: return img
    out_ = img.astype(np.float32); mu = out_[v].mean(0)
    g = np.exp(np.clip(rng.normal(0, 0.28), -0.5, 0.5) + rng.normal(0, 0.04, 3)); b = rng.normal(0, 5, 3)
    out_[v] = np.clip((out_[v] - mu) * g + mu + b, 0, 255)
    return out_.round().astype(np.uint8)


def haze_global(img, rng):
    s = rng.uniform(0.1, 0.35)
    o = img.astype(np.float32) * (1 - s) + np.array([180, 190, 225], np.float32) * s; o[~valid_of(img)] = 0
    return restretch(o.round().astype(np.uint8))


def block_region(h, w, rng):
    """Mosaic block: half-plane (row/col cut) or a corner rectangle, like the seams seen in 06 / 11."""
    r = np.zeros((h, w), bool); k1, k2 = rng.integers(h // 5, 4 * h // 5, 2)
    m = rng.integers(3)
    if m == 0: r[:k1] = True
    elif m == 1: r[:, :k1] = True
    else: r[:k1, :k2] = True
    if rng.random() < .5: r = ~r
    return r


def haze_block(img, rng, strength=(0.1, 0.35), tint_r=(140, 180), tint_g=(160, 195), tint_b=(205, 245)):
    """Regional haze: blend one mosaic block toward a bluish veil, THEN stretch the whole tile.
    The clear block keeps its contrast, the hazy block keeps lifted blacks / low contrast (as in test_033/049/096).
    Default ranges = calibration sweep in (b) (closest to the real hazy-patch signature)."""
    reg = block_region(*img.shape[:2], rng) & valid_of(img)
    s = rng.uniform(*strength); tint = np.array([rng.uniform(*tint_r), rng.uniform(*tint_g), rng.uniform(*tint_b)], np.float32)
    o = img.astype(np.float32); o[reg] = o[reg] * (1 - s) + tint * s
    return restretch(o.round().astype(np.uint8))


def seam_aug(img, rng, bank=None):
    return restretch(contrast_aug(img, rng, block_region(*img.shape[:2], rng)))


# (a) no-op proof
d_glob = [np.abs(haze_global(x.copy(), rng).astype(int) - restretch(x.copy()).astype(int))[valid_of(x)].mean() for x in X[::6]]
d_ctrl = [np.abs(contrast_aug(x.copy(), rng).astype(int) - x.astype(int))[valid_of(x)].mean() for x in X[::6]]
d_blk = [np.abs(haze_block(x.copy(), rng).astype(int) - restretch(x.copy()).astype(int))[valid_of(x)].mean() for x in X[::6]]
print(f"(a) mean |aug - restretch(original)| per valid px: global haze+restretch {np.mean(d_glob):.2f} | "
      f"haze_block {np.mean(d_blk):.2f} | (contrast_aug vs original {np.mean(d_ctrl):.2f})")
print("    -> global haze + restretch is ~identity: v1/v2 'haze' never showed the model a hazy image.")

# (b) signature of a hazy block: dark floor (p2) and contrast (p98-p2) inside 128 patches, haze index from 11
def block_sig(img):
    a = np.ascontiguousarray(img); v = valid_of(a); hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV); r = []
    for y in range(0, 1024, 128):
        for x in range(0, 1024, 128):
            s = (slice(y, y + 128), slice(x, x + 128))
            if v[s].mean() < .95: continue
            px = a[s].reshape(-1, 3).astype(np.float32); gy = px.mean(1)
            r.append([np.percentile(gy, 2), np.percentile(gy, 98) - np.percentile(gy, 2), hsv[s][..., 1].mean(), (px[:, 2] - px[:, 0]).mean()])
    return np.array(r).reshape(-1, 4)
def sig_table(arr):
    return np.concatenate(Parallel(n_jobs=16)(delayed(block_sig)(a) for a in arr))
S_tr, S_te = sig_table(X), sig_table(T)
S_v2 = sig_table([haze_global(x.copy(), np.random.default_rng(i)) for i, x in enumerate(X)])
cols = ["dark_p2", "contrast", "sat", "blue_minus_red"]
hz = lambda S: S[(S[:, 0] > 45) & (S[:, 1] < 90)]  # 'lifted blacks + low contrast' = hazy veil
print("(b) share of 128-px patches that look hazy (dark floor >45 and contrast <90) and their signature:")
for n, S in (("train", S_tr), ("test", S_te), ("train+global haze (v2)", S_v2)):
    h = hz(S); print(f"    {n:44s} {len(h) / len(S):.3f}   {dict(zip(cols, h.mean(0).round(1)))}")
# calibration sweep (applied to 100% of tiles): target = the real hazy signature (train and test agree: floor ~62, contrast ~60, sat ~60-67, B-R ~23-25)
target = hz(S_te).mean(0)
sweep = [("s.25-.6 grey-blue (first try)", dict(strength=(0.25, 0.6), tint_r=(150, 200), tint_g=(165, 205), tint_b=(200, 240))),
         ("s.1-.35 grey-blue", dict(strength=(0.1, 0.35), tint_r=(150, 200), tint_g=(165, 205), tint_b=(200, 240))),
         ("s.1-.35 blue (default)", dict()),
         ("s.05-.25 blue", dict(strength=(0.05, 0.25)))]
for n, kw in sweep:
    S = sig_table([haze_block(x.copy(), np.random.default_rng(i), **kw) for i, x in enumerate(X)]); h = hz(S)
    print(f"    haze_block {n:32s} {len(h) / len(S):.3f}   {dict(zip(cols, h.mean(0).round(1)))}  L1 to test sig {np.abs(h.mean(0) - target).mean():.1f}")

# (c) visual grid
frac = Y.reshape(len(Y), -1).mean(1); valid = np.array([valid_of(x).mean() for x in X])
reps = {"sawah dominan": int(np.argmax(frac * (valid > .95))), "permukiman": int(np.argmin(frac + (valid < .95))),
        "campuran": int(np.argmin(abs(frac - .4) + (valid < .95))), "NoData besar": int(np.argmin(abs(valid - .5))), "seam": 73}
bank = load_gain_bank()
ops = {"original": None, "contrast": contrast_aug, "seam": seam_aug, "haze global (v2)": haze_global,
       "haze_block (v3)": haze_block, "cloud": lambda x, r: restretch(clouds(x, r, (0.05, 0.3), (0.3, 0.8)))}
fig, ax = plt.subplots(len(reps) + 1, len(ops), figsize=(3.3 * len(ops), 3.4 * (len(reps) + 1)))
for r, (name, i) in enumerate(reps.items()):
    for c, (op, f) in enumerate(ops.items()):
        im = X[i] if f is None else f(X[i].copy(), np.random.default_rng(r * 10 + c))
        ax[r, c].imshow(im[::4, ::4]); ax[r, c].contour(Y[i][::4, ::4], colors="lime", linewidths=.5)
        ax[r, c].set_title(f"train_{i:03d} {name} | {op}", fontsize=7); ax[r, c].axis("off")
for c, t in enumerate([33, 49, 96, 128, 114, 0]):
    ax[-1, c].imshow(T[t][::4, ::4]); ax[-1, c].set_title(f"REAL test_{t:03d} (reference)", fontsize=8); ax[-1, c].axis("off")
plt.tight_layout(); plt.savefig(out("14_aug_v3_grid.png"), dpi=85); plt.close()

# (d) coverage: adversarial AUC test vs augmented train (features exactly as in 11)
from common import patch_feats, FEAT
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def pipeline(img, rng, kind):
    if kind == "none": return img
    if rng.random() < .5: img = contrast_aug(img, rng)
    if kind == "v2":
        for p, f in ((.05, seam_aug), (.05, haze_global), (.05, lambda x, r: restretch(clouds(x, r, (0.05, 0.3), (0.3, 0.8))))):
            if rng.random() < p: img = f(img, rng)
    if kind == "v3":
        for p, f in ((.25, seam_aug), (.3, haze_block), (.1, lambda x, r: restretch(clouds(x, r, (0.05, 0.3), (0.3, 0.8))))):
            if rng.random() < p: img = f(img, rng)
    return img


def feats(arr, split, off):
    res = Parallel(n_jobs=16)(delayed(patch_feats)(a) for a in arr)
    return pd.DataFrame([[split, off + i, *r] for i, f in enumerate(res) for r in f], columns=["split", "tile", "py", "px", *FEAT])


import sys
Fte = feats(T, "test", 1000) if "--coverage" in sys.argv else None; rows = []
for kind in (("none", "v2", "v3") if "--coverage" in sys.argv else ()):
    aug = [pipeline(x.copy(), np.random.default_rng(1000 * rep + i), kind) for rep in range(2) for i, x in enumerate(X)]
    Ftr = feats(aug, "train", 0); Ftr["tile"] = Ftr.tile % len(X)
    Dd = pd.concat([Ftr, Fte], ignore_index=True); y = (Dd.split == "test").values.astype(int); oof = np.zeros(len(Dd))
    for tr, va in GroupKFold(5).split(Dd, y, Dd.tile):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31, subsample=.8, subsample_freq=1, colsample_bytree=.8, verbose=-1)
        oof[va] = m.fit(Dd.iloc[tr][FEAT], y[tr]).predict_proba(Dd.iloc[va][FEAT])[:, 1]
    # coverage: share of test patches that look 'train-like' (p_test < 0.5)
    rows.append(dict(pipeline=kind, adv_auc=roc_auc_score(y, oof), test_patches_covered=(oof[y == 1] < .5).mean()))
    print(rows[-1], flush=True)
if rows: pd.DataFrame(rows).to_csv(out("14_coverage.csv"), index=False)
print("saved 14_aug_v3_grid.png, 14_coverage.csv")

# (e) artifact check: small black pixels (dark px clipped to 0 by the per-tile stretch, NOT NoData) stayed black under
# clouds/haze because valid_of() treats every 0 px as NoData -> black speckles inside white clouds. Fix: NoData = large
# black components only (same rule as post-processing), so dark px get veiled like their neighbours.
def clouds_fixed(img, rng, cover=(0.05, 0.3), opacity=(0.3, 0.8)):
    from augs import fractal_noise
    h, w = img.shape[:2]; n = fractal_noise(h, w, rng); t = np.quantile(n, 1 - rng.uniform(*cover))
    alpha = cv2.GaussianBlur(np.clip((n - t) / (1 - t + 1e-6) * 2.5, 0, 1) * rng.uniform(*opacity), (0, 0), 6)[..., None]
    o = img.astype(np.float32) * (1 - alpha) + np.array(rng.uniform(215, 250, 3), np.float32) * alpha
    o[nodata_mask(img)] = 0
    return restretch(o.round().astype(np.uint8)), alpha[..., 0]
from augs import fractal_noise
rows = []
for i in range(0, len(X), 4):
    x = X[i]; nd = nodata_mask(x)
    a_old = restretch(clouds(x.copy(), np.random.default_rng(i), (0.05, 0.3), (0.3, 0.8)))
    a_new, alpha = clouds_fixed(x.copy(), np.random.default_rng(i))
    under = (alpha > .3) & ~nd
    if under.sum() < 1000: continue
    rows.append(dict(old=(a_old.max(-1)[under] == 0).mean(), new=(a_new.max(-1)[under] == 0).mean()))
r = pd.DataFrame(rows).mean()
print(f"(e) share of pure-black px under synthetic cloud (alpha>0.3, outside NoData): old {r.old:.4%} | fixed {r.new:.4%}")
i = 90; x = X[i]
a_old = restretch(clouds(x.copy(), np.random.default_rng(5), (0.2, 0.3), (0.6, 0.8))); a_new, _ = clouds_fixed(x.copy(), np.random.default_rng(5), (0.2, 0.3), (0.6, 0.8))
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for a_, im, t in zip(ax, (x, a_old, a_new), ("original", "cloud (old: speckles)", "cloud (fixed)")):
    a_.imshow(im[300:812, 500:1012]); a_.set_title(f"train_{i:03d} {t}"); a_.axis("off")
plt.tight_layout(); plt.savefig(out("14_cloud_fix.png"), dpi=90); plt.close()
