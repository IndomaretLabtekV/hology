"""EDA 07: does preprocessing help *on the metric*? Frozen ImageNet backbone -> linear probe, grouped CV, macro F1.
Variants: raw resized (text+border+both orientations), crop only, crop+CLAHE. Also: Normal-vs-Abn AUC per variant
(if raw >> crop on that, raw was riding the compression shortcut)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, timm
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from prep import preprocess, img_path, ROOT

OUT = ROOT / "outputs/07"; OUT.mkdir(parents=True, exist_ok=True)
g = pd.read_csv(ROOT / "outputs/05/groups.csv")
tr = g[g.label != "?"].reset_index(drop=True)
CLASSES = ["Normal", "Benign", "Malignant"]; y = tr.label.map(CLASSES.index).values

def raw(i):  # naive baseline: whole image, just resized (as most notebooks would do)
    return cv2.resize(np.asarray(Image.open(img_path(i)).convert("L")), (768, 1024), interpolation=cv2.INTER_AREA)
def crop_noclahe(i): return preprocess(img_path(i), clahe=False)
def crop_clahe(i): return cv2.imread(str(ROOT / "cache" / i.replace(".jpg", ".png")), 0)
def _steps(i): return preprocess(img_path(i), clahe=False, return_steps=True)[1]
def raw_flip(i):  # raw pixels (bg + text kept), orientation normalized only
    st = _steps(i); return cv2.resize(np.ascontiguousarray(st["trimmed"][:, ::-1] if st["flipped"] else st["trimmed"]), (768, 1024), interpolation=cv2.INTER_AREA)
def masked_fullcanvas(i):  # bg/text zeroed + flipped, but NO crop/zoom -> physical scale preserved
    st = _steps(i); return cv2.resize(np.ascontiguousarray(st["masked"]), (768, 1024), interpolation=cv2.INTER_AREA)
def raw_masked_noflip(i):  # bg/text zeroed, original orientation, no crop
    st = _steps(i); a = st["masked"]; a = a[:, ::-1] if st["flipped"] else a
    return cv2.resize(np.ascontiguousarray(a), (768, 1024), interpolation=cv2.INTER_AREA)
def bg_only(i):  # breast (dilated) zeroed -> only background/text/border left. Any skill here = acquisition fingerprint
    st = _steps(i); a = st["trimmed"].copy(); mk = cv2.dilate(st["mask"].astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    a[mk] = 0; a = a[:, ::-1] if st["flipped"] else a
    return cv2.resize(np.ascontiguousarray(a), (768, 1024), interpolation=cv2.INTER_AREA)
def masked_loose(i):  # looser mask (keeps faint skin line / periphery), still bg+text zeroed, no crop
    st = _steps(i); a = st["trimmed"]; mk = cv2.dilate(st["mask"].astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
    a = np.where(mk, a, 0).astype(np.uint8); a = a[:, ::-1] if st["flipped"] else a
    return cv2.resize(np.ascontiguousarray(a), (768, 1024), interpolation=cv2.INTER_AREA)
VARIANTS = {"bg_only": bg_only, "masked_loose": masked_loose, "raw": raw, "crop": crop_noclahe, "crop+clahe": crop_clahe, "raw_flip": raw_flip,
            "masked_fullcanvas": masked_fullcanvas, "raw_masked_noflip": raw_masked_noflip}

net = timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=0).eval()
MEAN, STD = torch.tensor([.485, .456, .406])[:, None, None], torch.tensor([.229, .224, .225])[:, None, None]
@torch.no_grad()
def feats(imgs):
    x = torch.stack([torch.tensor(cv2.resize(a, (384, 512)) / 255., dtype=torch.float32)[None].repeat(3, 1, 1) for a in imgs])
    return net((x - MEAN) / STD).numpy()

def cv_eval(X, seeds=5):
    f1s, aucs, P = [], [], np.zeros((len(y), 3))
    for s in range(seeds):
        p = np.zeros((len(y), 3))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, tr.group):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
            p[b] = clf.fit(X[a], y[a]).predict_proba(X[b])
        f1s.append(f1_score(y, p.argmax(1), average="macro")); aucs.append(roc_auc_score(y == 0, p[:, 0])); P += p / seeds
    return np.mean(f1s), np.std(f1s), np.mean(aucs), P

for name, fn in VARIANTS.items():
    cache = OUT / f"feat_{name.replace('+', '_')}.npy"
    if not cache.exists():
        with ProcessPoolExecutor(8) as ex: imgs = list(ex.map(fn, tr.image_id))
        np.save(cache, np.concatenate([feats(imgs[k:k + 16]) for k in range(0, len(imgs), 16)]))
    X = np.load(cache)
    f1, sd, auc, P = cv_eval(X)
    bm = y > 0
    print(f"{name:11s} macroF1={f1:.3f}±{sd:.3f}  Normal-vs-Abn AUC={auc:.3f}  Benign-vs-Malig AUC={roc_auc_score(y[bm] == 2, P[bm, 2] / (P[bm, 1] + P[bm, 2])):.3f}")
    print("   confusion (rows=true N/B/M):", confusion_matrix(y, P.argmax(1)).tolist())
