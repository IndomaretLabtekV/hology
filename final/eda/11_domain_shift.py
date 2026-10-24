"""11 - Train -> test domain shift, measured WITHOUT labels (test images only, no test labels / LB).
(a) Shape-normalised radial power spectrum per tile: is test blurrier / a different GSD? Which Gaussian sigma
    (or down-up rescale) applied to train reproduces the test spectrum?  -> calibrates blur/scale augmentation.
(b) Patch-level adversarial validation (LightGBM, GroupKFold by tile): AUC train-vs-test and which features differ.
    Same classifier for 'spatial val fold vs its train folds': does our CV see a shift as large as the test one?"""
import numpy as np, pandas as pd, matplotlib, cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from common import *
from common import patch_feats, FEAT

X, T = train_images(), test_images()
folds = pd.read_csv(out("03d_folds.csv"))
W, NB = 256, 40
fy, fx = np.meshgrid(np.fft.fftfreq(W), np.fft.fftfreq(W), indexing="ij")
RAD = np.sqrt(fx ** 2 + fy ** 2); BINS = np.linspace(0.01, 0.5, NB + 1); BI = np.digitize(RAD, BINS) - 1
HANN = np.outer(np.hanning(W), np.hanning(W)).astype(np.float32)
CENT = (BINS[1:] + BINS[:-1]) / 2


def psd(img, sigma=0.0, rescale=1.0):
    """Mean radial PSD over fully-valid 256 windows, normalised to unit total power (shape only, stretch-invariant)."""
    a = np.ascontiguousarray(img)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if sigma > 0: g = cv2.GaussianBlur(g, (0, 0), sigma)
    if rescale != 1.0:
        s = int(round(1024 * rescale)); g = cv2.resize(cv2.resize(g, (s, s), interpolation=cv2.INTER_AREA), (1024, 1024), interpolation=cv2.INTER_LINEAR)
    v = a.max(-1) > 0
    acc, n = np.zeros(NB), 0
    for y in range(0, 1024, W):
        for x in range(0, 1024, W):
            if v[y:y + W, x:x + W].mean() < 0.999: continue
            w = g[y:y + W, x:x + W]; w = (w - w.mean()) * HANN
            P = np.abs(np.fft.fft2(w)) ** 2
            r = np.bincount(BI[(BI >= 0) & (BI < NB)], P[(BI >= 0) & (BI < NB)], NB)
            acc += r / np.bincount(BI[(BI >= 0) & (BI < NB)], minlength=NB); n += 1
    if n == 0: return np.full(NB, np.nan)
    acc /= n; return np.log10(acc / acc.sum())


def all_psd(arr, **kw):
    return np.array(Parallel(n_jobs=16)(delayed(psd)(a, **kw) for a in arr))


P_tr, P_te = all_psd(X), all_psd(T)
ok_tr, ok_te = ~np.isnan(P_tr).any(1), ~np.isnan(P_te).any(1)
m_tr, m_te = np.nanmean(P_tr[ok_tr], 0), np.nanmean(P_te[ok_te], 0)
print(f"(a) tiles with >=1 fully valid 256 window: train {ok_tr.sum()}/{len(X)}, test {ok_te.sum()}/{len(T)}")
hi = CENT > 0.25
print("    test - train log10 PSD (shape-normalised): low f %.3f | mid f %.3f | high f (>0.25 cyc/px) %.3f"
      % ((m_te - m_tr)[CENT < 0.08].mean(), (m_te - m_tr)[(CENT >= 0.08) & ~hi].mean(), (m_te - m_tr)[hi].mean()))
# high-frequency energy ratio per tile -> distribution comparison
hf_tr, hf_te = P_tr[ok_tr][:, hi].mean(1), P_te[ok_te][:, hi].mean(1)
from scipy.stats import ks_2samp
print(f"    per-tile HF log-power: train {hf_tr.mean():.3f}+-{hf_tr.std():.3f}  test {hf_te.mean():.3f}+-{hf_te.std():.3f}  KS p={ks_2samp(hf_tr, hf_te).pvalue:.2g}")

# which degradation of TRAIN reproduces the TEST spectrum? (distance between mean curves)
sub = np.where(ok_tr)[0][::2]
cands = [("none", {}), *[(f"blur s{s}", {"sigma": s}) for s in (0.3, 0.5, 0.7, 1.0)],
         *[(f"rescale x{r}", {"rescale": r}) for r in (0.9, 0.8, 0.7)]]
rows = []
for name, kw in cands:
    m = np.nanmean(all_psd(X[sub], **kw), 0)
    rows.append(dict(cand=name, dist_to_test=np.abs(m - m_te).mean(), hf_shift=(m - m_te)[hi].mean(), curve=m))
cd = pd.DataFrame(rows)
print("    train degradation -> L1 distance to mean test spectrum (lower = matches test):")
print(cd[["cand", "dist_to_test", "hf_shift"]].round(4).to_string(index=False))
# per test tile: which blur sigma best matches it individually (spread = how wide the blur aug range must be)
sig_grid = np.array([0, 0.3, 0.5, 0.7, 1.0, 1.4])
curves = np.array([np.nanmean(all_psd(X[sub], sigma=s), 0) for s in sig_grid])
best_sig = sig_grid[np.argmin(np.abs(P_te[ok_te][:, None, hi] - curves[None, :, hi]).mean(-1), 1)]
best_sig_tr = sig_grid[np.argmin(np.abs(P_tr[ok_tr][:, None, hi] - curves[None, :, hi]).mean(-1), 1)]
print("    per-tile best-matching sigma, share of tiles:")
print(pd.DataFrame({"sigma": sig_grid, "train": [(best_sig_tr == s).mean() for s in sig_grid],
                    "test": [(best_sig == s).mean() for s in sig_grid]}).round(3).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(18, 5))
ax[0].plot(CENT, m_tr, label="train"); ax[0].plot(CENT, m_te, label="test"); ax[0].set_xlabel("cycles/px"); ax[0].set_ylabel("log10 norm. power"); ax[0].legend(); ax[0].set_title("Radial PSD (shape-normalised)")
for r in cd.itertuples(): ax[1].plot(CENT, r.curve - m_te, label=r.cand)
ax[1].axhline(0, c="k", lw=.8); ax[1].set_title("train(+degradation) - test"); ax[1].legend(fontsize=7); ax[1].set_xlabel("cycles/px")
ax[2].hist(hf_tr, 30, alpha=.6, density=True, label="train"); ax[2].hist(hf_te, 30, alpha=.6, density=True, label="test"); ax[2].set_title("per-tile HF log-power"); ax[2].legend()
plt.tight_layout(); plt.savefig(out("11_psd.png"), dpi=110); plt.close()


# (b) patch-level adversarial validation
def table(arr, split, offset=0):
    res = Parallel(n_jobs=16)(delayed(patch_feats)(a) for a in arr)
    rows = [[split, offset + i, *r] for i, f in enumerate(res) for r in f]
    return pd.DataFrame(rows, columns=["split", "tile", "py", "px", *FEAT])


D = pd.concat([table(X, "train"), table(T, "test", 1000)], ignore_index=True)
D.to_pickle(CACHE / "11_patch_feats.pkl")
print(f"\n(b) patches: train {(D.split == 'train').sum()}, test {(D.split == 'test').sum()}")


def adv_auc(df, ylab, feats=FEAT, n=5):
    y = ylab.values.astype(int); oof = np.zeros(len(df)); imp = np.zeros(len(feats))
    for tr, va in GroupKFold(n).split(df, y, df.tile):
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1).fit(df.iloc[tr][feats], y[tr])
        oof[va] = m.predict_proba(df.iloc[va][feats])[:, 1]; imp += m.booster_.feature_importance("gain")
    return roc_auc_score(y, oof), pd.Series(imp / imp.sum(), feats).sort_values(ascending=False), oof


auc, imp, oof_adv = adv_auc(D, D.split == "test")
print(f"    adversarial AUC train vs test (grouped by tile): {auc:.3f}")
print("    top features:", imp.head(6).round(3).to_dict())
sd = D.groupby("split")[FEAT].mean().T; sd["shift_std"] = (sd.test - sd.train) / D[D.split == "train"][FEAT].std()
print(sd.round(3).sort_values("shift_std").to_string())

# same detector for each spatial CV fold: val fold vs the rest of train
tr = D[D.split == "train"].copy(); tr["fold"] = folds.fold.values[tr.tile]
fold_auc = {k: adv_auc(tr, tr.fold == k)[0] for k in range(5)}
print("    adversarial AUC val-fold vs rest-of-train:", {k: round(v, 3) for k, v in fold_auc.items()},
      "-> compare with train-vs-test", round(auc, 3))
# where in the test does the shift sit? per test tile mean 'testness'
D["p_test"] = oof_adv
tile_testness = D[D.split == "test"].groupby("tile").p_test.mean()
tile_testness.to_csv(out("11_test_tile_testness.csv"))
D[D.split == "train"].groupby("tile").p_test.mean().to_csv(out("11_train_tile_testness.csv"))

fig, ax = plt.subplots(2, 4, figsize=(18, 8))
for a_, f in zip(ax.flat[:6], imp.index[:6]):
    a_.hist(D[D.split == "train"][f], 50, alpha=.6, density=True, label="train"); a_.hist(D[D.split == "test"][f], 50, alpha=.6, density=True, label="test")
    a_.set_title(f"{f} (gain {imp[f]:.2f})"); a_.legend(fontsize=7)
ax.flat[6].bar(range(5), fold_auc.values()); ax.flat[6].axhline(auc, c="r", label="train vs test"); ax.flat[6].set_ylim(.4, 1); ax.flat[6].set_title("adv AUC: val fold vs rest"); ax.flat[6].legend()
ax.flat[7].hist(D[D.split == "train"].p_test, 50, alpha=.6, density=True, label="train"); ax.flat[7].hist(D[D.split == "test"].p_test, 50, alpha=.6, density=True, label="test"); ax.flat[7].set_title("OOF p(test) per patch"); ax.flat[7].legend()
plt.tight_layout(); plt.savefig(out("11_adversarial.png"), dpi=110); plt.close()

# most / least test-like test tiles, for eyeballing what 'test style' is
order = tile_testness.sort_values().index
pick = list(order[-4:]) + list(order[:2]); trp = D[D.split == "train"].groupby("tile").p_test.mean().sort_values().index[-2:]
fig, ax = plt.subplots(2, 4, figsize=(20, 10))
for a_, t in zip(ax.flat, pick + list(trp)):
    img = T[t - 1000] if t >= 1000 else X[t]
    a_.imshow(img[::2, ::2]); a_.set_title(f"{'test_%03d' % (t - 1000) if t >= 1000 else 'train_%03d' % t}  p(test)={D[D.tile == t].p_test.mean():.2f}"); a_.axis("off")
plt.suptitle("4 most test-like test tiles | 2 least test-like test tiles | 2 most test-like TRAIN tiles")
plt.tight_layout(); plt.savefig(out("11_testness_tiles.png"), dpi=90); plt.close()
print("saved 11_psd.png, 11_adversarial.png, 11_testness_tiles.png")
