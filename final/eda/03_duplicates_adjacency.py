"""03 - Duplicates, overlaps and spatial adjacency.
(a) exact / near-duplicate tiles (pHash) inside train, inside test, train<->test
(b) partial overlap via phase correlation on downsampled tiles
(c) edge-continuity adjacency -> rebuild the train mosaic -> spatial groups for CV that mimic the X-axis split."""
import json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imagehash, cv2
from PIL import Image
from scipy.sparse.csgraph import connected_components
from common import *

X, Y, T = train_images(), train_masks(), test_images()
A = np.concatenate([X, T]); names = [f.stem for f in TRAIN_IMG] + [f.stem for f in TEST_IMG]
nT = len(X)

# (a) perceptual hash
ph = [imagehash.phash(Image.fromarray(a[::4, ::4])) for a in A]
H = np.array([[ph[i] - ph[j] for j in range(len(A))] for i in range(len(A))])
np.fill_diagonal(H, 99)
for nm, sl_i, sl_j in [("train-train", slice(0, nT), slice(0, nT)), ("test-test", slice(nT, None), slice(nT, None)),
                       ("train-test", slice(0, nT), slice(nT, None))]:
    sub = H[sl_i, sl_j]
    print(f"[pHash {nm}] min dist={sub.min()}  pairs<=6: {(sub <= 6).sum() // (2 if sl_i == sl_j else 1)}  pairs<=10: {(sub <= 10).sum() // (2 if sl_i == sl_j else 1)}")
md5 = pd.Series([hash(a.tobytes()) for a in A]); print("exact byte duplicates:", md5.duplicated().sum())

# (b) partial overlap via phase correlation (256px gray, nodata-masked)
G = np.stack([cv2.resize(cv2.cvtColor(np.ascontiguousarray(a), cv2.COLOR_RGB2GRAY), (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32) for a in A])
win = cv2.createHanningWindow((256, 256), cv2.CV_32F)
ov = []
for i in range(len(A)):
    for j in range(i + 1, len(A)):
        if (G[i] > 0).mean() < .2 or (G[j] > 0).mean() < .2: continue
        (dx, dy), r = cv2.phaseCorrelate(G[i], G[j], win)
        if r > 0.25 and (abs(dx) > 1 or abs(dy) > 1):
            ov.append((names[i], names[j], round(dx * 4), round(dy * 4), round(r, 3)))
print(f"\n[phase-corr overlap r>0.25, shift>4px] candidate pairs: {len(ov)}")
for o in sorted(ov, key=lambda t: -t[-1])[:10]: print("  ", o)

# (c) edge continuity: seam ratio = |i_edge - j_edge| / internal gradient. ~1 = adjacent.
def edge_cost(a, b, axis):
    if axis == 1: e1, e1b, e2, e2b = a[:, -1], a[:, -2], b[:, 0], b[:, 1]   # a left of b
    else:         e1, e1b, e2, e2b = a[-1], a[-2], b[0], b[1]               # a above b
    e1, e1b, e2, e2b = [e.astype(np.float32) for e in (e1, e1b, e2, e2b)]
    valid = (e1.max(-1) > 0) & (e2.max(-1) > 0)
    if valid.sum() < 128: return np.inf, valid.sum()
    cross = np.abs(e1 - e2)[valid].mean()
    inner = 0.5 * (np.abs(e1 - e1b)[valid].mean() + np.abs(e2 - e2b)[valid].mean())
    return cross / (inner + 1e-3), valid.sum()

links = []
for axis, tag in [(1, "R"), (0, "D")]:
    C = np.full((len(A), len(A)), np.inf)
    for i in range(len(A)):
        for j in range(len(A)):
            if i != j: C[i, j] = edge_cost(A[i], A[j], axis)[0]
    for i in range(len(A)):
        j = int(np.argmin(C[i]))
        # mutual best + absolute threshold
        if C[i, j] < 1.6 and int(np.argmin(C[:, j])) == i:
            links.append((i, j, tag, float(C[i, j])))
costs = np.array([l[3] for l in links])
print(f"\n[edge adjacency] mutual-best links: {len(links)}  cost p50={np.median(costs):.2f} max={costs.max():.2f}")
cross_links = [(names[i], names[j], t) for i, j, t, c in links if (i < nT) != (j < nT)]
print("train<->test adjacency links (should be ~0 because buffer tiles were removed):", len(cross_links), cross_links[:5])

# lay out components on a grid
adj = np.zeros((len(A), len(A)), bool)
for i, j, _, _ in links: adj[i, j] = adj[j, i] = True
ncomp, lab = connected_components(adj, directed=False)
pos = {}
for c in range(ncomp):
    mem = np.where(lab == c)[0]; root = mem[0]; pos[root] = (0, 0); stack = [root]
    while stack:
        u = stack.pop()
        for i, j, t, _ in links:
            for a, b, sgn in ((i, j, 1), (j, i, -1)):
                if a == u and b not in pos:
                    dx, dy = (sgn, 0) if t == "R" else (0, sgn)
                    pos[b] = (pos[u][0] + dx, pos[u][1] + dy); stack.append(b)
sizes = np.bincount(lab)
print(f"components: {ncomp}; size dist (train-containing): ",
      sorted([int(sizes[c]) for c in set(lab[:nT])], reverse=True)[:15])
grid_conflicts = sum(len(v) - len(set(v)) for v in [[pos[m] for m in np.where(lab == c)[0]] for c in range(ncomp)])
print("grid-position conflicts (two tiles same cell; >0 means bad links):", grid_conflicts)

df = pd.DataFrame({"id": names, "split": ["train"] * nT + ["test"] * len(T), "comp": lab,
                   "gx": [pos[i][0] for i in range(len(A))], "gy": [pos[i][1] for i in range(len(A))]})
df.to_csv(out("03_tile_layout.csv"), index=False)

# render the biggest train components as mosaics
big = [c for c in np.argsort(-sizes) if (lab[np.where(lab == c)[0]] < 9e9).all() and sizes[c] >= 4][:4]
fig, axs = plt.subplots(1, len(big), figsize=(8 * len(big), 8))
for ax, c in zip(np.atleast_1d(axs), big):
    mem = np.where(lab == c)[0]; gx = np.array([pos[m][0] for m in mem]); gy = np.array([pos[m][1] for m in mem])
    s = 128; W, Hh = (gx.max() - gx.min() + 1) * s, (gy.max() - gy.min() + 1) * s
    canvas = np.full((Hh, W, 3), 255, np.uint8)
    for m, x, y in zip(mem, gx - gx.min(), gy - gy.min()):
        t = cv2.resize(np.ascontiguousarray(A[m]), (s, s), interpolation=cv2.INTER_AREA)
        if m < nT:
            mk = cv2.resize(Y[m], (s, s), interpolation=cv2.INTER_NEAREST).astype(bool); t[mk] = (0.6 * t[mk] + [0, 100, 0]).astype(np.uint8)
        canvas[y * s:(y + 1) * s, x * s:(x + 1) * s] = t
    ax.imshow(canvas); ax.set_title(f"comp {c}: {len(mem)} tiles ({(mem < nT).sum()} train) green=sawah"); ax.axis("off")
plt.tight_layout(); plt.savefig(out("03_mosaics.png"), dpi=80); plt.close()

plt.figure(figsize=(6, 4)); plt.hist(costs, bins=30); plt.xlabel("edge seam ratio"); plt.title("accepted adjacency link costs")
plt.savefig(out("03_link_costs.png"), dpi=100); plt.close()
json.dump({"overlaps": ov}, open(out("03_overlaps.json"), "w"))
