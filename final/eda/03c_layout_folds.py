"""03c - Merge overlap (stride-512) + edge adjacency into one global layout (half-tile units),
measure label agreement inside overlaps (= annotation noise ceiling), and build spatial CV folds."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from scipy.sparse.csgraph import connected_components
from common import *

X, Y, T = train_images(), train_masks(), test_images()
A = np.concatenate([X, T]); names = [f.stem for f in TRAIN_IMG] + [f.stem for f in TEST_IMG]; nT = len(X)
idx = {n: i for i, n in enumerate(names)}
ov = pd.read_csv(out("03b_overlaps.csv"))
lay = pd.read_csv(out("03_tile_layout.csv"))

# edges in half-tile units: content of b at (x,y) == content of a at (x+dx,y+dy) -> b sits at a + (dx,dy)
E = []
for r in ov.itertuples():
    if r.d4 != 0: continue
    E.append((idx[r.a], idx[r.b], int(round(r.dx / 512)), int(round(r.dy / 512)), "ov"))
# 03 links: 'R' = a left of b (b = a + (2,0)), 'D' = a above b (b = a + (0,2))
# recompute from layout grid of 03 (comp-relative coords, 1 unit = 1024px)
for c, g in lay.groupby("comp"):
    if len(g) < 2: continue
    ids = g.index.values
    for i in ids:
        for j in ids:
            dx, dy = lay.gx[j] - lay.gx[i], lay.gy[j] - lay.gy[i]
            if (abs(dx) + abs(dy)) == 1: E.append((i, j, 2 * dx, 2 * dy, "edge"))

print("overlap edges:", sum(e[4] == "ov" for e in E), "| adjacency edges:", sum(e[4] == "edge" for e in E))
print("overlap shift vectors (half-tile units):", pd.Series([(e[2], e[3]) for e in E if e[4] == "ov"]).value_counts().to_dict())

adj = np.zeros((len(A), len(A)), bool)
for i, j, *_ in E: adj[i, j] = adj[j, i] = True
ncomp, lab = connected_components(adj, directed=False)
pos = {}
conflicts = 0
for c in range(ncomp):
    mem = np.where(lab == c)[0]; pos[mem[0]] = (0, 0); st = [mem[0]]
    while st:
        u = st.pop()
        for i, j, dx, dy, _ in E:
            for a, b, s in ((i, j, 1), (j, i, -1)):
                if a != u: continue
                p = (pos[u][0] + s * dx, pos[u][1] + s * dy)
                if b not in pos: pos[b] = p; st.append(b)
                elif pos[b] != p: conflicts += 1
print("layout conflicts (inconsistent edges):", conflicts)

df = pd.DataFrame({"id": names, "split": ["train"] * nT + ["test"] * len(T), "comp": lab,
                   "hx": [pos[i][0] for i in range(len(A))], "hy": [pos[i][1] for i in range(len(A))]})
tr = df[df.split == "train"]
cs = tr.groupby("comp").size().sort_values(ascending=False)
print(f"train components: {len(cs)}  sizes: {cs.values.tolist()}")
te = df[df.split == "test"]; print(f"test components: {te.comp.nunique()}  sizes: {te.groupby('comp').size().sort_values(ascending=False).values.tolist()}")

# effective number of unique pixels in train (overlaps double count)
unique_area = 0
for c, g in tr.groupby("comp"):
    cells = set()
    for r in g.itertuples():
        for ox in (0, 1):
            for oy in (0, 1): cells.add((r.hx + ox, r.hy + oy))
    unique_area += len(cells) / 4
print(f"train tiles={nT}, unique-area equivalent tiles={unique_area:.1f} (overlap redundancy x{nT / unique_area:.2f})")

# label agreement inside overlaps (same ground, two label crops) + image agreement
agree = []
for r in ov[ov.kind == "train-train"].itertuples():
    i, j = idx[r.a], idx[r.b]; dx, dy = int(r.dx), int(r.dy)
    x0, x1 = max(0, -dx), min(1024, 1024 - dx); y0, y1 = max(0, -dy), min(1024, 1024 - dy)
    mi = Y[i][y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(bool); mj = Y[j][y0:y1, x0:x1].astype(bool)
    xi = A[i][y0 + dy:y1 + dy, x0 + dx:x1 + dx].astype(np.int16); xj = A[j][y0:y1, x0:x1].astype(np.int16)
    inter, union = (mi & mj).sum(), (mi | mj).sum()
    agree.append((r.a, r.b, inter / max(union, 1), (mi != mj).mean(), np.abs(xi - xj).mean(), union))
ag = pd.DataFrame(agree, columns=["a", "b", "iou", "disagree_px", "img_mae", "union"])
print("\n== label agreement in train overlaps ==")
print("pairs:", len(ag), "| micro IoU across overlaps: %.4f" % ((ag.iou * ag.union).sum() / ag.union.sum()))
print(ag[["iou", "disagree_px", "img_mae"]].describe().round(4).to_string())
print("worst label agreement:\n", ag.sort_values("iou").head(6).to_string(index=False))
ag.to_csv(out("03c_overlap_label_agreement.csv"), index=False)

# spatial folds: group = component, balance by tile count and sawah pixels, order by mean hx inside big comps
K = 5
frac = Y.reshape(nT, -1).mean(1)
tr = tr.assign(frac=frac)
comp_stats = tr.groupby("comp").agg(n=("id", "size"), saw=("frac", "sum")).sort_values("n", ascending=False)
fold_n, fold_s, comp_fold = np.zeros(K), np.zeros(K), {}
for c, r in comp_stats.iterrows():
    k = int(np.argmin(fold_n + 3 * fold_s / max(comp_stats.saw.sum(), 1) * comp_stats.n.sum() / K))
    comp_fold[c] = k; fold_n[k] += r.n; fold_s[k] += r.saw
tr["fold"] = tr.comp.map(comp_fold)
print("\nfolds -> tiles:", tr.groupby("fold").size().tolist(), " mean sawah frac:", tr.groupby("fold").frac.mean().round(3).tolist())
# sanity: no overlap pair crosses folds
f = dict(zip(tr.id, tr.fold))
cross = [(r.a, r.b) for r in ov[ov.kind == "train-train"].itertuples() if f[r.a] != f[r.b]]
print("overlap pairs crossing folds (must be 0):", len(cross))
tr[["id", "comp", "hx", "hy", "fold", "frac"]].to_csv(out("03c_folds.csv"), index=False)
df.to_csv(out("03c_layout.csv"), index=False)

# render largest train components with fold colours
big = cs.index[:3]
fig, axs = plt.subplots(1, len(big), figsize=(9 * len(big), 9))
for ax, c in zip(axs, big):
    g = df[df.comp == c]; s = 128
    W = (g.hx.max() - g.hx.min() + 2) * s // 2; Hh = (g.hy.max() - g.hy.min() + 2) * s // 2
    canvas = np.full((Hh, W, 3), 255, np.uint8)
    for r in g.itertuples():
        x, y = (r.hx - g.hx.min()) * s // 2, (r.hy - g.hy.min()) * s // 2
        t = cv2.resize(np.ascontiguousarray(A[r.Index]), (s, s), interpolation=cv2.INTER_AREA)
        valid = t.max(-1) > 0
        canvas[y:y + s, x:x + s][valid] = t[valid]
    for r in g.itertuples():
        x, y = (r.hx - g.hx.min()) * s // 2, (r.hy - g.hy.min()) * s // 2
        ax.add_patch(plt.Rectangle((x, y), s, s, fill=False, ec="yellow", lw=.6))
    ax.imshow(canvas); ax.set_title(f"comp {c}: {len(g)} tiles (stride 512, yellow=tile footprint)"); ax.axis("off")
plt.tight_layout(); plt.savefig(out("03c_layout.png"), dpi=70); plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(ag.iou, bins=30, color="tab:green"); ax[0].set_title("label IoU inside overlapping train pairs")
ax[1].scatter(ag.img_mae, ag.iou, s=10); ax[1].set_xlabel("image MAE in overlap"); ax[1].set_ylabel("label IoU")
plt.tight_layout(); plt.savefig(out("03c_overlap_agreement.png"), dpi=100); plt.close()
