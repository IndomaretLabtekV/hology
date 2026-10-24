"""17 - 'Tiny prediction -> empty mask' rule for the per-image (Airbus-style) metric found in 16.
Per image: empty GT + empty pred = 1.0, empty GT + any pred = 0.0. A tile whose predicted paddy covers < EMPTY of the
tile almost surely scores 0 when GT is non-empty (IoU of a tiny blob < 0.5), so emptying it is ~free; if GT is
empty it gains a full image (+1/71 public). Cutoff chosen on OOF only (cost must be 0 for every run).
(a) OOF cost per run/cutoff, (b) apply to submission CSVs (RLE decode -> rule -> RLE encode), (c) show the emptied tiles.
usage: python 17_empty_rule_submit.py <sub1.csv> [sub2.csv ...]"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

R = ROOT / "results"; Y = train_masks(); X = train_images(); S = 2
Ys = Y[:, ::S, ::S].astype(bool); NODs = np.stack([nodata_mask(x)[::S, ::S] for x in X])
TH = np.arange(0.5, 0.96, 0.05)


def score(m, g):
    if g.sum() == 0: return float(m.sum() == 0)
    if m.sum() == 0: return 0.0
    return float(((m & g).sum() / (m | g).sum() > TH).mean())


print("(a) OOF: images emptied and score change per cutoff (thr .15)")
oofs = {k: np.load(R / d / "oof.npy", mmap_mode="r")[:, ::S, ::S].astype(np.float32) for k, d in
        (("run1", "paddy_dinov3"), ("v2", "paddy_dinov3_v2"), ("v3", "paddy_dinov3_v3"))}
oofs["run1+v3"] = (oofs["run1"] + oofs["v3"]) / 2
for k, P in oofs.items():
    ms = [(p > .15) & ~nd for p, nd in zip(P, NODs)]; base = np.array([score(m, g) for m, g in zip(ms, Ys)]); fr = np.array([m.mean() for m in ms])
    for e in (.005, .01, .015, .02):
        hit = fr < e
        print(f"  {k:8s} <{e:.3f}: emptied {hit.sum()} | their score before {base[hit].round(1).tolist()} | delta {-base[hit].sum() / len(base):+.4f}")
EMPTY = 0.01  # largest cutoff with zero OOF cost for every run (see table above)


def rle_decode(rle, shape=(1024, 1024)):
    m = np.zeros(shape[0] * shape[1], np.uint8)
    if isinstance(rle, str) and rle:
        s = np.array(rle.split(), int)
        for a, l in zip(s[0::2], s[1::2]): m[a - 1:a - 1 + l] = 1
    return m.reshape(shape, order="F")


def rle_encode(mask):
    p = np.concatenate([[0], mask.flatten(order="F").astype(np.uint8), [0]])
    runs = np.where(p[1:] != p[:-1])[0] + 1; runs[1::2] -= runs[::2]
    return " ".join(map(str, runs))


assert rle_encode(rle_decode("3 4 10 2")) == "3 4 10 2"
T = test_images(); shown = []
for f in map(Path, sys.argv[1:]):
    sub = pd.read_csv(f, keep_default_na=False); fr = []
    for i, r in sub.iterrows():
        m = rle_decode(r.EncodedPixels); fr.append(m.mean())
        if 0 < m.mean() < EMPTY: sub.at[i, "EncodedPixels"] = ""; shown.append((r.ImageId, f.stem, m))
    o = f.with_name(f.stem + "_empty1.csv"); sub.to_csv(o, index=False)
    fr = np.array(fr)
    print(f"(b) {f.name}: emptied {[sub.ImageId[i] for i in np.where((fr > 0) & (fr < EMPTY))[0]]} | already empty {(fr == 0).sum()} | tiles <5%: {(fr < .05).sum()} -> {o.name}")
if shown:
    fig, ax = plt.subplots(1, len(shown[:6]), figsize=(5 * len(shown[:6]), 5), squeeze=False)
    for a_, (iid, src, m) in zip(ax[0], shown[:6]):
        a_.imshow(T[int(iid[-3:])][::2, ::2]); a_.contour(m[::2, ::2], [.5], colors="yellow"); a_.set_title(f"{iid} ({src}) pred {m.mean():.3%} -> empty", fontsize=8); a_.axis("off")
    plt.tight_layout(); plt.savefig(out("17_emptied_tiles.png"), dpi=80); plt.close()
    print("saved 17_emptied_tiles.png")
