"""EDA 14: resolution vs JPEG-shortcut trade-off for the Mammo-FM input (1520x912).
Draft 1/4 (1185x885) must be UPSAMPLED to 1520 (no real detail gain); draft 1/2 (2370x1770) is downsampled (real detail)
but may re-expose the Normal-only high-quality JPEG signature. Measure both:
 (a) low-level feature probe AUC (shortcut), (b) high-frequency energy retained (detail)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT, img_path
from prep_fm import fm_variant

OUT = ROOT / "outputs/14"; OUT.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(ROOT / "data/train.csv"); C = ["Normal", "Benign", "Malignant"]; y = tr.label.map(C.index).values
NAMES = ["noise_std", "lap_std", "blockiness", "mean", "std", "p5", "p50", "p95", "n_levels", "hf_energy", "bg_std", "bg_levels", "bg_res"]


def feats(a):
    m = a > 0; a = a.astype(np.float32)
    res = a - cv2.medianBlur(a.astype(np.uint8), 3).astype(np.float32)
    lap = cv2.Laplacian(a, cv2.CV_32F)
    dx = np.abs(np.diff(a, axis=1)); blk = dx[:, 7::8].mean() / (dx.mean() + 1e-6)
    v = a[m]
    # high-frequency energy: fraction of spectrum power above 1/4 Nyquist, inside a central tissue patch
    h, w = a.shape; patch = a[h // 3:2 * h // 3, :w // 3]
    F = np.abs(np.fft.fftshift(np.fft.fft2(patch - patch.mean()))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(patch.shape[0]))[:, None] * 2; fx = np.fft.fftshift(np.fft.fftfreq(patch.shape[1]))[None] * 2
    r = np.sqrt(fy ** 2 + fx ** 2)
    return [res[m].std(), lap[m].std(), blk, v.mean(), v.std(), *np.percentile(v, [5, 50, 95]), len(np.unique(v.astype(np.uint8))),
            F[r > 0.25].sum() / F.sum()]


def bg_feats(b):
    # background-only image (breast zeroed): JPEG quality leaves a different noise texture in the dark background
    v = b[(b > 0) & (b < 60)].astype(np.float32)
    res = (b.astype(np.float32) - cv2.medianBlur(b, 3).astype(np.float32))[(b > 0) & (b < 60)]
    return [v.std() if v.size else 0, len(np.unique(v)) if v.size else 0, res.std() if res.size else 0]


def job(args):
    p = img_path(args[0])
    return feats(fm_variant(p, "v2_bbox", scale=args[1])) + bg_feats(fm_variant(p, "bg_only", scale=args[1]))


cv = StratifiedKFold(5, shuffle=True, random_state=0)
clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
res = {}
for sc in [4, 2, 1]:
    with ProcessPoolExecutor(16) as ex:
        X = pd.DataFrame(list(ex.map(job, [(i, sc) for i in tr.image_id])), columns=NAMES)
    res[sc] = X
    for task, yy, sub in [("Normal vs Abn", (y == 0).astype(int), np.ones(len(y), bool)), ("Benign vs Malig", (y == 2).astype(int), y > 0)]:
        p = cross_val_predict(clf, X[NAMES][sub], yy[sub], cv=cv, method="predict_proba")[:, 1]
        per = {f: round(max(roc_auc_score(yy[sub], X[f][sub]), 1 - roc_auc_score(yy[sub], X[f][sub])), 2) for f in ["blockiness", "noise_std", "hf_energy", "bg_std", "bg_levels", "bg_res"]}
        print(f"draft 1/{sc}  {task:16s} all-feature AUC={roc_auc_score(yy[sub], p):.3f}  {per}")
    print(f"draft 1/{sc}  mean hf_energy={X.hf_energy.mean():.4f}  noise_std={X.noise_std.mean():.2f}\n")

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k, f in enumerate(["bg_res", "noise_std", "hf_energy"]):
    for sc, X in res.items():
        for lab, ls in [(0, "-"), (1, "--")]:
            ax[k].hist(X[f][(y == 0) == (lab == 0)], bins=30, histtype="step", ls=ls, label=f"1/{sc} {'Normal' if lab == 0 else 'Abn'}")
    ax[k].set_title(f)
ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(OUT / "resolution_shortcut.png", dpi=80)

# visual: zoomed tissue crop at each scale (does 1/2 show more real detail?)
i = tr[tr.label == "Malignant"].image_id.iloc[3]
fig, ax = plt.subplots(1, 3, figsize=(12, 5))
for a, sc in zip(ax, [4, 2, 1]):
    im = fm_variant(img_path(i), "v2_bbox", scale=sc); a.imshow(im[600:900, 150:450], cmap="gray"); a.set_title(f"{i} draft 1/{sc} (zoom)")
plt.tight_layout(); plt.savefig(OUT / "zoom_detail.png", dpi=80)
