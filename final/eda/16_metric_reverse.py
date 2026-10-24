"""16 - The LB metric is NOT pixel micro-IoU. Every public score (ours and the top teams') is an exact multiple of
1/710 = 71 public images x 10 levels -> per-image score in steps of 0.1, i.e. Airbus-style: for each image,
mean over IoU thresholds 0.50:0.05:0.95 of [IoU > t] (empty GT & empty pred = 1, empty GT & any pred = 0).
(a) proof table (LB score * 710), (b) OOF score of run1 / v2 / v3 / ensembles under that metric vs micro-IoU:
does the OOF ordering now match the LB ordering? (c) post-processing tuned on OOF under the per-image metric:
threshold, min component area, 'tiny prediction -> empty' rule. (d) apply to test, write submissions.
No LB probing: every choice below is made on OOF only."""
import sys
import numpy as np, pandas as pd, matplotlib, cv2
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

LB = {"run1 5f @.10": .76338, "run1 5f @.15": .75633, "v2 5f @.10": .73943, "v2 5f @.15": .75070, "v3 5f @.10": .74647,
      "v3 5f @.15": .74647, "run1+v3 @.10": .75211, "run1+v3 @.15": .76056, "run1+v2 @.15": .75915, "top1": .80422, "top3": .80000}
print("(a) LB * 710 (integer => 71 images x 10 levels):")
print(pd.Series({k: round(v * 710, 3) for k, v in LB.items()}).to_string())

R = ROOT / "results"
RUNS = {"run1": R / "paddy_dinov3", "v2": R / "paddy_dinov3_v2", "v3": R / "paddy_dinov3_v3"}
Y = train_masks(); X = train_images(); N = len(Y)
NOD = np.stack([nodata_mask(x) for x in X])
S = 2  # evaluate at half resolution (IoU of 512x512 polygons ~ identical, 4x faster)
Ys, NODs = Y[:, ::S, ::S].astype(bool), NOD[:, ::S, ::S]
OOF = {k: np.load(p / "oof.npy", mmap_mode="r")[:, ::S, ::S].astype(np.float32) for k, p in RUNS.items()}
OOF["run1+v3"] = (OOF["run1"] + OOF["v3"]) / 2
OOF["run1+v2+v3"] = (OOF["run1"] + OOF["v2"] + OOF["v3"]) / 3
TH = np.arange(0.5, 0.96, 0.05)


def img_score(pr, g):
    if g.sum() == 0: return float(pr.sum() == 0)
    if pr.sum() == 0: return 0.0
    iou = (pr & g).sum() / (pr | g).sum()
    return float((iou > TH).mean())


def post(p, nd, thr, min_area=0, empty_below=0):
    m = (p > thr) & ~nd
    if min_area:
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        keep = st[:, cv2.CC_STAT_AREA] >= min_area / S ** 2; keep[0] = False; m = keep[lab]
    if empty_below and m.mean() < empty_below: m[:] = False
    return m


def evaluate(P, thr, **kw):
    sc, tp, un = [], 0, 0
    for p, g, nd in zip(P, Ys, NODs):
        m = post(p, nd, thr, **kw); sc.append(img_score(m, g)); tp += (m & g).sum(); un += (m | g).sum()
    return np.mean(sc), tp / un, np.array(sc)


print("\n(b) OOF: per-image Airbus score vs micro-IoU")
rows = []
for k, P in OOF.items():
    for t in (.1, .15, .3, .5):
        a, mi, _ = evaluate(P, t); rows.append(dict(run=k, thr=t, airbus=a, micro=mi))
B = pd.DataFrame(rows); print(B.pivot(index="run", columns="thr", values="airbus").round(4).to_string())
print(B.pivot(index="run", columns="thr", values="micro").round(4).add_prefix("micro@").to_string())

# where is the score lost? per-image score vs paddy fraction
_, _, sc = evaluate(OOF["run1+v3"], .15)
frac = Ys.reshape(N, -1).mean(1)
bins = pd.cut(frac, [0, .05, .15, .35, .6, 1])
print("\nrun1+v3 @.15 per-image score by GT paddy fraction:"); print(pd.DataFrame({"score": sc, "bin": bins}).groupby("bin", observed=True).score.agg(["mean", "count"]).round(3).to_string())
print("share of images scoring 0:", (sc == 0).mean().round(3))

# (c) post-processing grid on OOF (per-image metric), on the ensemble and per run
print("\n(c) post-processing grid on OOF (per-image score):")
grid = []
for k in ("run1", "v3", "run1+v3", "run1+v2+v3"):
    for t in (.1, .15, .2, .3, .4, .5):
        for ma in (0, 2048, 8192, 32768):
            a, mi, _ = evaluate(OOF[k], t, min_area=ma); grid.append(dict(run=k, thr=t, min_area=ma, airbus=a, micro=mi))
G = pd.DataFrame(grid); G.to_csv(out("16_postproc_grid.csv"), index=False)
for k, g in G.groupby("run"):
    b = g.loc[g.airbus.idxmax()]; base = g[(g.thr == .15) & (g.min_area == 0)].airbus.iloc[0]
    print(f"  {k:11s} best thr {b.thr} min_area {int(b.min_area)} -> {b.airbus:.4f} (thr .15 no pp {base:.4f}) micro {b.micro:.4f}")
print(G[G.run == "run1+v3"].pivot(index="thr", columns="min_area", values="airbus").round(4).to_string())

# tiny prediction -> empty: cost on train (no empty GT in train => pure cost) and how many test tiles it touches
tp_test = {k: np.load(p / "test_prob_mean.npy", mmap_mode="r") for k, p in RUNS.items()}
T = test_images()
ens_te = (tp_test["run1"].astype(np.float32) + tp_test["v3"].astype(np.float32)) / 2
nod_te = np.stack([nodata_mask(t) for t in T])
pf_te = np.array([((p > .15) & ~nd).mean() for p, nd in zip(ens_te, nod_te)])
pf_tr = np.array([((p > .15) & ~nd).mean() for p, nd in zip(OOF["run1+v3"], NODs)])
print("\npredicted paddy fraction <1%/<2%/<5%: train OOF", [(pf_tr < x).sum() for x in (.01, .02, .05)], "| test", [(pf_te < x).sum() for x in (.01, .02, .05)])
for e in (.005, .01, .02):
    a, _, s2 = evaluate(OOF["run1+v3"], .15, empty_below=e)
    print(f"  empty rule <{e:.3f}: OOF score {a:.4f} (images affected {(pf_tr < e).sum()}, their score before {sc[pf_tr < e].round(2).tolist()})")
print("GT paddy frac train min %.4f; test tiles with tiny predictions (frac<2%%):" % frac.min(), [f"test_{i:03d}:{pf_te[i]:.3f}" for i in np.where(pf_te < .02)[0]])

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for k in OOF:
    g = B[B.run == k]; ax[0].plot(g.thr, g.airbus, "o-", label=k)
ax[0].set_title("OOF per-image Airbus score vs threshold"); ax[0].legend(); ax[0].set_xlabel("thr")
ax[1].scatter(frac, sc, s=12); ax[1].set_xlabel("GT paddy fraction"); ax[1].set_ylabel("image score"); ax[1].set_title("run1+v3 @.15 per image")
plt.tight_layout(); plt.savefig(out("16_metric.png"), dpi=100); plt.close()
np.save(CACHE / "16_ens_te.npy", pf_te)
print("saved 16_metric.png, 16_postproc_grid.csv")
