"""03b - Verify partial overlaps pixel-wise (incl. D4 flips/rotations).
Phase-correlation on gradient images (nodata-robust) proposes a shift; the pair is accepted only if the
overlapping valid pixels agree (NCC>0.9 on full-res gray) over >=3% of the tile."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from joblib import Parallel, delayed
from common import *

X, T = train_images(), test_images()
A = np.concatenate([X, T]); names = [f.stem for f in TRAIN_IMG] + [f.stem for f in TEST_IMG]; nT = len(X)
S = 256; f = 1024 // S

def prep(a):
    g = cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY).astype(np.float32)
    valid = a.max(-1) > 0
    g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA); v = cv2.resize(valid.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
    gm = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    gm[~cv2.erode(v.astype(np.uint8), np.ones((5, 5))).astype(bool)] = 0
    return g, v, gm

P = [prep(a) for a in A]
D4 = [lambda z: z, lambda z: np.rot90(z, 1), lambda z: np.rot90(z, 2), lambda z: np.rot90(z, 3),
      lambda z: z[:, ::-1], lambda z: np.rot90(z[:, ::-1], 1), lambda z: np.rot90(z[:, ::-1], 2), lambda z: np.rot90(z[:, ::-1], 3)]
win = cv2.createHanningWindow((S, S), cv2.CV_32F)

def ncc_at(gi, vi, gj, vj, dx, dy):
    # content of j at (x,y) == content of i at (x+dx, y+dy)
    H = W = gi.shape[0]
    x0, x1 = max(0, -dx), min(W, W - dx); y0, y1 = max(0, -dy), min(H, H - dy)
    if x1 - x0 < 8 or y1 - y0 < 8: return 0, 0
    a = gi[y0 + dy:y1 + dy, x0 + dx:x1 + dx]; b = gj[y0:y1, x0:x1]
    m = vi[y0 + dy:y1 + dy, x0 + dx:x1 + dx] & vj[y0:y1, x0:x1]
    if m.sum() < 0.03 * H * W: return 0, m.mean()
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6)), m.sum() / (H * W)

def check(i):
    res = []
    gi, vi, mi = P[i]
    for j in range(i + 1, len(A)):
        for k, t in enumerate(D4):
            gj, vj, mj = [np.ascontiguousarray(t(z)) for z in P[j]]
            (sx, sy), r = cv2.phaseCorrelate(mi, mj.astype(np.float32), win)
            dx, dy = -int(round(sx)), -int(round(sy))
            if k == 0 and abs(dx) <= 1 and abs(dy) <= 1: continue
            for ddx, ddy in ((dx, dy), (-dx, -dy)):
                c, ovf = ncc_at(gi, vi, gj, vj, ddx, ddy)
                if c > 0.9:
                    res.append((names[i], names[j], k, ddx * f, ddy * f, round(c, 3), round(ovf, 3))); break
    return res

out_rows = sum(Parallel(n_jobs=-1)(delayed(check)(i) for i in range(len(A))), [])
df = pd.DataFrame(out_rows, columns=["a", "b", "d4", "dx", "dy", "ncc", "overlap"])
df["kind"] = [("train" if a.startswith("train") else "test") + "-" + ("train" if b.startswith("train") else "test") for a, b in zip(df.a, df.b)]
df.to_csv(out("03b_overlaps.csv"), index=False)
print("verified overlapping pairs (NCC>0.9, overlap>=3%):", len(df))
print(df.groupby("kind").size().to_string())
print("D4 transform distribution (0=identity):", df.d4.value_counts().to_dict())
print("overlap fraction stats:\n", df.overlap.describe().round(3).to_string())
print(df.sort_values("overlap", ascending=False).head(25).to_string(index=False))

# visual proof for top pairs per kind
rows = pd.concat([g.sort_values("overlap", ascending=False).head(2) for _, g in df.groupby("kind")]) if len(df) else df
if len(rows):
    fig, ax = plt.subplots(len(rows), 3, figsize=(15, 5 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, (_, row) in enumerate(rows.iterrows()):
        i, j = names.index(row.a), names.index(row.b)
        bj = np.ascontiguousarray(D4[row.d4](A[j]))
        ax[r, 0].imshow(A[i]); ax[r, 0].set_title(row.a)
        ax[r, 1].imshow(bj); ax[r, 1].set_title(f"{row.b} (D4={row.d4})")
        M = np.float32([[1, 0, row.dx], [0, 1, row.dy]])
        warped = cv2.warpAffine(bj, M, (1024, 1024))
        blend = (0.5 * A[i] + 0.5 * warped).astype(np.uint8)
        ax[r, 2].imshow(blend); ax[r, 2].set_title(f"blend after shift dx={row.dx},dy={row.dy} ncc={row.ncc} ov={row.overlap}")
        for a in ax[r]: a.axis("off")
    plt.tight_layout(); plt.savefig(out("03b_overlap_proof.png"), dpi=60); plt.close()
