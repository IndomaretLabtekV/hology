"""EDA 15: which augmentations are label-preserving for Mammo-FM? (no training)
Protocol: logistic regression fit on CLEAN Mammo-FM features (grouped by OCR patient pairs), evaluated on
augmented validation images. A realistic aug keeps predictions (high agreement, small F1 drop, high cosine);
a destructive aug (black wedge, lifted background, extreme shear) flips them.
Augs = Mammo-FM's own pretraining recipe (hflip, vflip, affine rot20/shear20/scale0.8-1.2, elastic) vs our fixes."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT, img_path
from prep_fm import load_fm_encoder, fm_features, fm_variant

torch.set_num_threads(22)
OUT = ROOT / "outputs/15"; OUT.mkdir(parents=True, exist_ok=True)
PREP = "fm_notext"
pairs = pd.read_csv(ROOT / "outputs/13/pairs.csv")[["image_id", "pair"]]
tr = pd.read_csv(ROOT / "data/train.csv").merge(pairs, on="image_id")
C = ["Normal", "Benign", "Malignant"]; y = tr.label.map(C.index).values
g_order = pd.read_csv(ROOT / "outputs/05/groups.csv"); g_order = g_order[g_order.label != "?"].image_id.tolist()
Xc = pd.DataFrame(np.load(ROOT / f"outputs/12/feat_mammo_fm_{PREP}_s2.npy"), index=g_order).loc[tr.image_id].values
rng = np.random.default_rng(0)


def affine(a, rot=0, shear=0, scale=1.0, tx=0.0, ty=0.0, border=cv2.BORDER_CONSTANT):
    h, w = a.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
    S = np.array([[1, np.tan(np.deg2rad(shear)), 0], [0, 1, 0]], np.float64); S[0, 2] = -S[0, 1] * h / 2
    M = (np.vstack([M, [0, 0, 1]]) @ np.vstack([S, [0, 0, 1]]))[:2]; M[:, 2] += [tx * w, ty * h]
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=0)


def elastic(a, alpha=10, sigma=15):
    h, w = a.shape
    dx = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x, y_ = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(a, x + dx, y_ + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def jitter(a):  # torchvision-style ColorJitter(0.3, 0.3) on uint8
    f = a.astype(np.float32) * rng.uniform(.7, 1.3); c = rng.uniform(.7, 1.3)
    return np.clip(c * f + (1 - c) * f.mean(), 0, 255).astype(np.uint8)


def masked_gamma(a):
    f = a.astype(np.float32) / 255.; g = rng.uniform(.7, 1.4)
    return np.where(a > 0, np.clip(f ** g * rng.uniform(.85, 1.15), 0, 1) * 255, 0).astype(np.uint8)


U = lambda lo, hi: float(rng.uniform(lo, hi))
AUGS = {
    "hflip": lambda a: a[:, ::-1],
    "vflip": lambda a: a[::-1],
    "rot20_const (FM)": lambda a: affine(a, rot=U(-20, 20)),
    "rot20_reflect": lambda a: affine(a, rot=U(-20, 20), border=cv2.BORDER_REFLECT_101),
    "shear20 (FM)": lambda a: affine(a, shear=U(-20, 20)),
    "scale0.8-1.2 (FM)": lambda a: affine(a, scale=U(.8, 1.2)),
    "affine_full (FM)": lambda a: affine(a, rot=U(-20, 20), shear=U(-20, 20), scale=U(.8, 1.2), tx=U(-.1, .1), ty=U(-.1, .1)),
    "affine_mild_reflect": lambda a: affine(a, rot=U(-10, 10), shear=U(-5, 5), scale=U(.9, 1.1), tx=U(-.03, .03), ty=U(-.03, .03), border=cv2.BORDER_REFLECT_101),
    "elastic (FM)": elastic,
    "colorjitter": jitter,
    "masked_gamma": masked_gamma,
}


def job(i):
    return fm_variant(img_path(i), PREP, 2)

with ProcessPoolExecutor(12) as ex:
    clean = list(ex.map(job, tr.image_id))
fm = load_fm_encoder()

folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(Xc, y, tr.pair))
P_clean = np.zeros((len(y), 3)); clfs = []
for a, b in folds:
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced")).fit(Xc[a], y[a])
    P_clean[b] = clf.predict_proba(Xc[b]); clfs.append((b, clf))
f1_clean = f1_score(y, P_clean.argmax(1), average="macro")
print(f"clean OOF macro F1 (pair-grouped) = {f1_clean:.3f}")

rows = []
for name, fn in AUGS.items():
    t0 = time.time()
    Xa = fm_features(fm, [np.ascontiguousarray(fn(a)) for a in clean])
    P = np.zeros_like(P_clean)
    for b, clf in clfs: P[b] = clf.predict_proba(Xa[b])
    cos = np.sum(Xa * Xc, 1) / (np.linalg.norm(Xa, axis=1) * np.linalg.norm(Xc, axis=1))
    rows.append(dict(aug=name, f1_aug=round(f1_score(y, P.argmax(1), average="macro"), 3), f1_drop=round(f1_clean - f1_score(y, P.argmax(1), average="macro"), 3),
                     pred_agree=round((P.argmax(1) == P_clean.argmax(1)).mean(), 3), cos=round(cos.mean(), 3), secs=int(time.time() - t0)))
    print(rows[-1], flush=True)
res = pd.DataFrame(rows).sort_values("pred_agree", ascending=False); res.to_csv(OUT / "aug_consistency.csv", index=False)
print("\n", res.to_string(index=False))

pk = [tr[tr.label == c].index[2] for c in C]
fig, ax = plt.subplots(3, 1 + len(AUGS), figsize=(2.1 * (1 + len(AUGS)), 11))
for r, k in enumerate(pk):
    ax[r, 0].imshow(clean[k], cmap="gray", vmin=0, vmax=255); ax[r, 0].set_title(f"{C[r]} clean", fontsize=8)
    for c, (name, fn) in enumerate(AUGS.items(), 1):
        ax[r, c].imshow(fn(clean[k]), cmap="gray", vmin=0, vmax=255); ax[r, c].set_title(name, fontsize=7)
for a in ax.flat: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "aug_samples.png", dpi=50)
