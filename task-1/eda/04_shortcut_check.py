"""EDA 04: can low-level (non-anatomical) features predict the label? If yes -> shortcut the CNN will exploit.
Features computed on (a) RAW full-res pixels and (b) the preprocessed cache. Grouped by nothing (patients unknown here),
so AUC is optimistic but the *comparison* raw vs cache is what matters."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from prep import img_path, ROOT

OUT = ROOT / "outputs/04"; OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(ROOT / "outputs/03/prep_stats.csv")


def lowlevel(a, mask):
    a = a.astype(np.float32)
    res = a - cv2.medianBlur(a.astype(np.uint8), 3).astype(np.float32)    # sensor/compression noise
    lap = cv2.Laplacian(a, cv2.CV_32F)
    # JPEG blockiness: mean |diff| across 8px boundaries vs inside blocks
    dx = np.abs(np.diff(a, axis=1))
    blk = dx[:, 7::8].mean() / (dx.mean() + 1e-6)
    v = a[mask]
    return [res[mask].std(), lap[mask].std(), blk, v.mean(), v.std(), *np.percentile(v, [5, 50, 95]),
            len(np.unique(a[mask].astype(np.uint8)))]  # n gray levels (bit depth / windowing trace)

NAMES = ["noise_std", "lap_std", "blockiness", "mean", "std", "p5", "p50", "p95", "n_levels"]


def job(iid):
    raw = np.asarray(Image.open(img_path(iid)).convert("L"))[1000:3000, 200:1800]  # full-res patch, mostly tissue side
    rm = raw > 20
    c = cv2.imread(str(ROOT / "cache" / iid.replace(".jpg", ".png")), 0)
    return lowlevel(raw, rm if rm.mean() > .05 else np.ones_like(rm)), lowlevel(c, c > 0)

with ProcessPoolExecutor(16) as ex:
    out = list(ex.map(job, m.image_id))
R = pd.DataFrame([o[0] for o in out], columns=NAMES); C = pd.DataFrame([o[1] for o in out], columns=NAMES)
for X in (R, C): X["label"] = m.label.values; X["mode"] = pd.read_csv(ROOT / "outputs/01/stats.csv").set_index("image_id").loc[m.image_id, "mode"].values
R.to_csv(OUT / "raw_feats.csv", index=False); C.to_csv(OUT / "cache_feats.csv", index=False)

tr = m.split.values == "train"
cv = StratifiedKFold(5, shuffle=True, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
print("AUC of a linear model on low-level features (1.0 = perfect shortcut, 0.5 = none)")
for name, X in [("RAW", R), ("CACHE", C)]:
    Xt, lab = X[tr], X.label[tr]
    for task, y, sub in [("Normal vs Abn", (lab == "Normal").astype(int), slice(None)),
                         ("Benign vs Malig", (lab == "Malignant").astype(int), lab != "Normal")]:
        Xs, ys = Xt[NAMES][sub], y[sub]
        p = cross_val_predict(clf, Xs, ys, cv=cv, method="predict_proba")[:, 1]
        per = {f: round(max(roc_auc_score(ys, Xs[f]), 1 - roc_auc_score(ys, Xs[f])), 2) for f in NAMES}
        print(f"  {name:5s} {task:16s} AUC={roc_auc_score(ys, p):.3f}   per-feature: {per}")

fig, ax = plt.subplots(2, len(NAMES), figsize=(3 * len(NAMES), 6))
for i, (name, X) in enumerate([("RAW", R), ("CACHE", C)]):
    for j, f in enumerate(NAMES):
        for lab, g in X.groupby("label"):
            ax[i, j].hist(g[f], bins=25, alpha=.5, label=lab, density=True)
        ax[i, j].set_title(f"{name} {f}", fontsize=9)
ax[0, 0].legend(); plt.tight_layout(); plt.savefig(OUT / "lowlevel_hist.png", dpi=70)
print("\ntest-set feature means (compare to train classes):\n", C.groupby("label")[NAMES].mean().round(2))
