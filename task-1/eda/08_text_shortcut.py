"""EDA 08: why does RAW beat CROP on Benign-vs-Malignant? Hypothesis: burned-in text (font size / style / box)
differs by export batch = by class. Measure non-breast bright blobs (text) and correlate with label."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from prep import load_gray, trim_border, breast_mask, img_path, ROOT

OUT = ROOT / "outputs/08"; OUT.mkdir(parents=True, exist_ok=True)
g = pd.read_csv(ROOT / "outputs/05/groups.csv")

def text_feats(i):
    a = trim_border(load_gray(img_path(i)))
    bm = breast_mask(a)
    t = ((a > 150) & ~cv2.dilate(bm.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(t, 8)
    st = st[1:]; st = st[(st[:, cv2.CC_STAT_AREA] > 3) & (st[:, cv2.CC_STAT_AREA] < 5000)]
    h = st[:, cv2.CC_STAT_HEIGHT] if len(st) else np.array([0])
    return dict(image_id=i, n_blobs=len(st), text_area=int(st[:, cv2.CC_STAT_AREA].sum()) if len(st) else 0,
                char_h_med=float(np.median(h)), char_h_max=int(h.max()),
                text_mean=float(a[t > 0].mean()) if t.any() else 0)

with ProcessPoolExecutor(16) as ex:
    f = pd.DataFrame(list(ex.map(text_feats, g.image_id)))
f = g.merge(f); f.to_csv(OUT / "text_feats.csv", index=False)
print(f.groupby("label")[["n_blobs", "text_area", "char_h_med", "char_h_max", "text_mean"]].describe().T.round(1).to_string())
bm = f[f.label.isin(["Benign", "Malignant"])]
nb = f[f.label != "?"]
for c in ["n_blobs", "text_area", "char_h_med", "char_h_max", "text_mean"]:
    a1 = roc_auc_score(bm.label == "Malignant", bm[c]); a2 = roc_auc_score(nb.label == "Normal", nb[c])
    print(f"{c:11s} AUC Benign-vs-Malig={max(a1, 1 - a1):.3f}  Normal-vs-Abn={max(a2, 1 - a2):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for l, d in f.groupby("label"):
    ax[0].hist(d.char_h_med, bins=30, alpha=.5, label=l); ax[1].hist(d.text_area, bins=30, alpha=.5, label=l)
ax[0].set_title("median text char height (px @1/4 res)"); ax[1].set_title("text area"); ax[0].legend()
plt.savefig(OUT / "text_hist.png", dpi=70)
