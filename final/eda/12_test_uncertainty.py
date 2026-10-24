"""12 - Where is the model unsure on TEST, and why? (run-1 GPU model: 5-fold mean test probs + OOF probs).
Uses only model outputs on unlabeled test images (no labels, no LB) -> diagnosis, not training signal.
(a) per-tile 'uncertain mass' (0.1 < p < 0.5): OOF train vs test. That band is exactly what thr 0.5 -> 0.1 flips.
(b) per-patch: which image properties (haze/seam/colour/texture, from 11) carry the uncertain mass in test?
(c) k-means 'appearance clusters' fit on train+test patches: per cluster train share, test share, GT paddy rate,
    OOF prob, test prob -> clusters frequent in test but rare in train with low test confidence = failure mode.
(d) object scale: connected-component size of predicted paddy in test vs OOF (same model -> comparable)."""
import numpy as np, pandas as pd, matplotlib, cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from common import *

R = ROOT / "results" / "paddy_dinov3"
X, Y, T = train_images(), train_masks(), test_images()
OOF = np.load(R / "oof.npy", mmap_mode="r"); TP = np.load(R / "test_prob_mean.npy", mmap_mode="r")
D = pd.read_pickle(CACHE / "11_patch_feats.pkl")
FEAT = ["mR", "mG", "mB", "sR", "sG", "sB", "satm", "sats", "valm", "grad", "lap", "lap_s2", "hf_ratio", "sat255", "dark"]
nod_tr = [nodata_mask(x) for x in X]; nod_te = [nodata_mask(t) for t in T]

# (a) per tile
def tile_stats(P, nods):
    r = []
    for p, nd in zip(P, nods):
        v = ~nd; p = p.astype(np.float32)[v]
        r.append(dict(unc=((p > .1) & (p < .5)).mean(), f10=(p > .1).mean(), f50=(p > .5).mean(), pm=p.mean()))
    return pd.DataFrame(r)
st_tr, st_te = tile_stats(OOF, nod_tr), tile_stats(TP, nod_te)
st_tr["gtr"] = [y[~nd].mean() for y, nd in zip(Y, nod_tr)]
print("(a) uncertain band 0.1<p<0.5 (share of valid px): OOF train %.4f | test %.4f" % (st_tr.unc.mean(), st_te.unc.mean()))
print("    pred frac p>0.5: train OOF %.3f (GT %.3f) | test %.3f;  p>0.1: train %.3f | test %.3f"
      % (st_tr.f50.mean(), st_tr.gtr.mean(), st_te.f50.mean(), st_tr.f10.mean(), st_te.f10.mean()))
print("    tiles with unc>5%%: train %d/%d, test %d/%d" % ((st_tr.unc > .05).sum(), len(st_tr), (st_te.unc > .05).sum(), len(st_te)))
# in train, what is the GT rate inside the uncertain band? (tells whether lowering thr there is right on train)
inb, gtb = 0, 0
for p, y, nd in zip(OOF, Y, nod_tr):
    b = (p > .1) & (p < .5) & ~nd; inb += b.sum(); gtb += y[b].sum()
print(f"    train OOF: GT paddy rate inside 0.1<p<0.5 band = {gtb / inb:.3f}  (LB says on test it must be >0.5 to pay off)")

# (b) per patch
P = 128
def patch_probs(prob, rows):
    p = prob.astype(np.float32)
    return [(((w := p[y:y + P, x:x + P]) > .1) & (w < .5)).mean() for y, x in rows] , [p[y:y + P, x:x + P].mean() for y, x in rows]
D["unc"], D["pm"], D["gt"] = np.nan, np.nan, np.nan
for t, g in D.groupby("tile"):
    rows = list(zip(g.py, g.px))
    if t >= 1000: u, m = patch_probs(TP[t - 1000], rows)
    else:
        u, m = patch_probs(OOF[t], rows); D.loc[g.index, "gt"] = [Y[t][y:y + P, x:x + P].mean() for y, x in rows]
    D.loc[g.index, "unc"], D.loc[g.index, "pm"] = u, m
# haze index: bright + desaturated + blue cast; seam flag per tile from 06
D["haze"] = (D.valm - D.satm) / 255 + (D.mB - D.mR) / 255
sc = pd.read_csv(out("06_seams_clouds.csv")); sc["tile"] = np.where(sc.split == "test", 1000 + sc.i, sc.i)
D = D.merge(sc[["tile", "n_seams"]], on="tile", how="left")
te, tr = D[D.split == "test"], D[D.split == "train"]
print("\n(b) test patches: share of total uncertain mass by haze-index quintile (quintile edges from train):")
q = np.quantile(tr.haze, [0, .2, .4, .6, .8, 1]); q[0], q[-1] = -9, 9
for name, d in (("train", tr), ("test", te)):
    b = pd.cut(d.haze, q, labels=False)
    print(f"   {name:5s} patch share per quintile {np.bincount(b, minlength=5) / len(d)}  mean unc {d.groupby(b).unc.mean().round(3).values}")
print("   corr(feature, unc) in test patches:")
print(te[FEAT + ["haze"]].corrwith(te.unc).sort_values().round(3).to_string())

# (c) appearance clusters
Z = StandardScaler().fit(D[FEAT]).transform(D[FEAT])
km = KMeans(12, n_init=4, random_state=0).fit(Z); D["cl"] = km.labels_
te, tr = D[D.split == "test"], D[D.split == "train"]
C = pd.DataFrame({"train_share": tr.cl.value_counts(normalize=True), "test_share": te.cl.value_counts(normalize=True),
                  "gt_rate": tr.groupby("cl").gt.mean(), "oof_p": tr.groupby("cl").pm.mean(), "oof_unc": tr.groupby("cl").unc.mean(),
                  "test_p": te.groupby("cl").pm.mean(), "test_unc": te.groupby("cl").unc.mean(), "haze": D.groupby("cl").haze.mean()}).fillna(0)
C["ratio_test/train"] = C.test_share / C.train_share.clip(1e-3)
C["unc_mass_test"] = C.test_share * C.test_unc / (C.test_share * C.test_unc).sum()
print("\n(c) appearance clusters (sorted by share of test uncertain mass):")
print(C.sort_values("unc_mass_test", ascending=False).round(3).to_string())
C.to_csv(out("12_clusters.csv"))

top = C.sort_values("unc_mass_test", ascending=False).index[:4]
fig, ax = plt.subplots(len(top) * 2, 6, figsize=(15, 5.2 * len(top)))
rng = np.random.default_rng(0)
for r, c in enumerate(top):
    for k, (name, d, src) in enumerate((("test", te, T), ("train", tr, X))):
        g = d[d.cl == c]
        for j in range(6):
            a_ = ax[2 * r + k, j]; a_.axis("off")
            if len(g) <= j: continue
            s = g.iloc[rng.integers(len(g))]; t = int(s.tile) - (1000 if name == "test" else 0)
            a_.imshow(src[t][int(s.py):int(s.py) + P, int(s.px):int(s.px) + P])
            a_.set_title(f"cl{c} {name} unc={s.unc:.2f} p={s.pm:.2f}" + ("" if name == "test" else f" gt={s['gt']:.2f}"), fontsize=7)
plt.tight_layout(); plt.savefig(out("12_cluster_examples.png"), dpi=90); plt.close()

# per-tile view: most uncertain test tiles with the prob map
worst = st_te.unc.sort_values().index[-6:][::-1]
fig, ax = plt.subplots(2, 6, figsize=(24, 8.5))
for j, i in enumerate(worst):
    ax[0, j].imshow(T[i][::2, ::2]); ax[0, j].set_title(f"test_{i:03d} unc={st_te.unc[i]:.2f}"); ax[0, j].axis("off")
    im = ax[1, j].imshow(TP[i][::2, ::2].astype(np.float32), cmap="RdYlGn", vmin=0, vmax=1); ax[1, j].axis("off")
    ax[1, j].contour(TP[i][::2, ::2].astype(np.float32), [0.1, 0.5], colors=["b", "k"], linewidths=.6)
ax[1, 0].set_title("prob (blue contour 0.1, black 0.5)")
plt.tight_layout(); plt.savefig(out("12_uncertain_test_tiles.png"), dpi=80); plt.close()

# (d) object scale (component area of predicted paddy at thr 0.3, full res)
def comp_areas(P, nods, thr=.3):
    a = []
    for p, nd in zip(P, nods):
        n, lab, st, _ = cv2.connectedComponentsWithStats(((p > thr) & ~nd).astype(np.uint8), connectivity=8)
        a += list(st[1:, cv2.CC_STAT_AREA])
    return np.array(a)
a_tr, a_te = comp_areas(OOF, nod_tr), comp_areas(TP, nod_te)
print("\n(d) predicted component area percentiles [25,50,75,95]: OOF train", np.percentile(a_tr, [25, 50, 75, 95]).astype(int),
      "| test", np.percentile(a_te, [25, 50, 75, 95]).astype(int))
# hole structure (texture scale proxy): median distance-to-boundary inside predicted paddy
def width(P, nods, thr=.3):
    w = []
    for p, nd in zip(P[::3], nods[::3]):
        m = ((p > thr) & ~nd).astype(np.uint8)
        if m.sum(): w.append(np.median(cv2.distanceTransform(m, cv2.DIST_L2, 5)[m > 0]))
    return np.array(w)
print("    median inner distance-to-edge (px): OOF train %.1f | test %.1f" % (np.median(width(OOF, nod_tr)), np.median(width(TP, nod_te))))
st_tr.to_csv(out("12_tile_unc_train.csv")); st_te.to_csv(out("12_tile_unc_test.csv"))
print("saved 12_cluster_examples.png, 12_uncertain_test_tiles.png, 12_clusters.csv")
