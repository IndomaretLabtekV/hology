"""04b - Hypothesis: every tile was exported with its own per-tile (per-channel or global) percentile stretch.
If true: valid-pixel percentiles p_lo/p_hi land on ~fixed values (0/255) for every tile, and the gain between
two overlapping tiles is predictable from their content-percentiles. That tells us exactly how to augment
(re-stretch random windows) and how to normalise (per-tile robust stretch is then nearly a no-op)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

X, T = train_images(), test_images()
A = np.concatenate([X, T]); names = [f.stem for f in TRAIN_IMG] + [f.stem for f in TEST_IMG]; idx = {n: i for i, n in enumerate(names)}
qs = [0.1, 0.5, 1, 2, 98, 99, 99.5, 99.9]
rows = []
for n, a in zip(names, A):
    v = a.max(-1) > 0
    px = a[v]
    r = {"id": n, "valid": v.mean()}
    for c, ch in enumerate("RGB"):
        p = np.percentile(px[:, c], qs)
        for q, val in zip(qs, p): r[f"{ch}_p{q}"] = val
        r[f"{ch}_n0"] = (px[:, c] == 0).mean(); r[f"{ch}_n255"] = (px[:, c] == 255).mean()
    rows.append(r)
P = pd.DataFrame(rows); P.to_csv(out("04b_percentiles.csv"), index=False)
cols = [f"{c}_p{q}" for c in "RGB" for q in qs]
print(P[[c for c in cols if any(s in c for s in ("p0.5", "p2", "p98", "p99.5"))]].describe().round(1).T.to_string())
print("\nfraction of valid px saturated at 255 per channel (median over tiles):", P[["R_n255", "G_n255", "B_n255"]].median().round(4).tolist())
print("fraction of valid px clipped at 0 per channel (median):", P[["R_n0", "G_n0", "B_n0"]].median().round(4).tolist())

# predict overlap gains from percentile ranges: if tile = stretch(raw, lo, hi) -> gain_ab = (hi_a-lo_a)/(hi_b-lo_b) in raw units
# we cannot see raw, but we can test the weaker implication: the overlap gain correlates with (content-range of a) / (content-range of b)
ph = pd.read_csv(out("04_overlap_photometric.csv"))
ph["valid_a"] = [P.valid[idx[a]] for a in ph.a]; ph["valid_b"] = [P.valid[idx[b]] for b in ph.b]
print("\ncorr(gain_G, valid_b-valid_a):", round(np.corrcoef(ph.gain_G, ph.valid_b - ph.valid_a)[0, 1], 3))

# simulate: take an overlap pair a,b. Use a's pixels; crop the region of tile b, stretch it to b's own p-lo/p-hi -> compare with b
ov = pd.read_csv(out("03b_overlaps.csv")); ov = ov[ov.d4 == 0]
sim = []
for r in ov.itertuples():
    i, j = idx[r.a], idx[r.b]; dx, dy = int(r.dx), int(r.dy)
    x0, x1 = max(0, -dx), min(1024, 1024 - dx); y0, y1 = max(0, -dy), min(1024, 1024 - dy)
    a = A[i][y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.float32); b = A[j][y0:y1, x0:x1].astype(np.float32)
    v = (a.max(-1) > 0) & (b.max(-1) > 0)
    if v.sum() < 5000: continue
    # best per-channel-independent vs shared-gain affine
    res = {}
    for mode in ("per_channel", "shared_gain"):
        if mode == "per_channel":
            pred = np.stack([np.polyval(np.polyfit(a[..., c][v], b[..., c][v], 1), a[..., c]) for c in range(3)], -1)
        else:
            g = np.polyfit(a[v].ravel(), b[v].ravel(), 1)[0]
            pred = np.stack([g * a[..., c] + (b[..., c][v] - g * a[..., c][v]).mean() for c in range(3)], -1)
        res[mode] = np.abs(pred - b)[v].mean()
    sim.append((r.a, r.b, res["per_channel"], res["shared_gain"], np.abs(a - b)[v].mean()))
S = pd.DataFrame(sim, columns=["a", "b", "mae_affine_perch", "mae_affine_shared", "mae_raw"])
print("\nresidual MAE after: per-channel affine %.2f | shared gain + per-ch offset %.2f | raw %.2f" % tuple(S[["mae_affine_perch", "mae_affine_shared", "mae_raw"]].median()))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
for c, col in zip("RGB", ("r", "g", "b")):
    ax[0].hist(P[f"{c}_p0.5"], bins=40, alpha=.5, color=col, label=c); ax[1].hist(P[f"{c}_p99.5"], bins=40, alpha=.5, color=col, label=c)
ax[0].set_title("per-tile p0.5 of valid pixels"); ax[1].set_title("per-tile p99.5 of valid pixels"); ax[0].legend()
ax[2].scatter(P.valid, P["G_p99.5"] - P["G_p0.5"], s=8); ax[2].set_xlabel("valid fraction"); ax[2].set_ylabel("G p99.5-p0.5")
ax[2].set_title("dynamic range vs valid area")
plt.tight_layout(); plt.savefig(out("04b_percentiles.png"), dpi=100); plt.close()
