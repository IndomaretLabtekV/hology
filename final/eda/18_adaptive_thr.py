"""18 - Per-image metric: low-paddy tiles score worst (0.33-0.41 OOF). Does a tile-adaptive threshold help on OOF?
Rule: base thr t0; if the tile's predicted paddy fraction at t0 is < F, use thr t1 for that tile. Grid on OOF only."""
import numpy as np, pandas as pd
from common import *
R = ROOT / "results"; S = 2; X = train_images()
Ys = train_masks()[:, ::S, ::S].astype(bool); NODs = np.stack([nodata_mask(x)[::S, ::S] for x in X])
TH = np.arange(0.5, 0.96, 0.05)
P = (np.load(R / "paddy_dinov3/oof.npy", mmap_mode="r")[:, ::S, ::S].astype(np.float32) + np.load(R / "paddy_dinov3_v3/oof.npy", mmap_mode="r")[:, ::S, ::S].astype(np.float32)) / 2
def score(m, g):
    if m.sum() == 0: return 0.0
    return float(((m & g).sum() / (m | g).sum() > TH).mean())
rows = []
for t0 in (.15,):
    for F in (0, .05, .1, .15, .25):
        for t1 in (.05, .15, .3, .5, .7):
            sc = []
            for p, g, nd in zip(P, Ys, NODs):
                m = (p > t0) & ~nd
                if m.mean() < F: m = (p > t1) & ~nd
                if m.mean() < .01: m[:] = False
                sc.append(score(m, g))
            rows.append(dict(t0=t0, F=F, t1=t1, score=np.mean(sc)))
D = pd.DataFrame(rows); print(D.pivot(index="F", columns="t1", values="score").round(4).to_string())
D.to_csv(out("18_adaptive_thr.csv"), index=False)
