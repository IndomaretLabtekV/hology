"""EDA 12: does Mammo-FM beat ImageNet, and which preprocessing does it want?
Frozen features -> logistic regression, StratifiedGroupKFold x5 seeds (same protocol as EDA 07).
No training of the backbone, only forward passes."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, timm, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT, img_path
from prep_fm import load_fm_encoder, fm_features, fm_variant, FM_H, FM_W

torch.set_num_threads(22)
OUT = ROOT / "outputs/12"; OUT.mkdir(parents=True, exist_ok=True)
g = pd.read_csv(ROOT / "outputs/05/groups.csv"); tr = g[g.label != "?"].reset_index(drop=True)  # row order = saved features
tr["group"] = tr[["image_id"]].merge(pd.read_csv(ROOT / "outputs/16/train_pairs_fm.csv"), on="image_id").pair_fm.values  # EDA 16 pairs
C = ["Normal", "Benign", "Malignant"]; y = tr.label.map(C.index).values


def probe(X, seeds=5):
    f1s, P = [], np.zeros((len(y), 3))
    for s in range(seeds):
        p = np.zeros((len(y), 3))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, tr.group):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
            p[b] = clf.fit(X[a], y[a]).predict_proba(X[b])
        f1s.append(f1_score(y, p.argmax(1), average="macro")); P += p / seeds
    bm = y > 0
    return dict(macro_f1=round(np.mean(f1s), 3), sd=round(np.std(f1s), 3), auc_normal=round(roc_auc_score(y == 0, P[:, 0]), 3),
                auc_BM=round(roc_auc_score(y[bm] == 2, P[bm, 2] / (P[bm, 1] + P[bm, 2])), 3))


def job(args):
    return fm_variant(img_path(args[0]), args[1], scale=args[2], harmonize=bool(args[3]))


def images(variant, scale, h):
    with ProcessPoolExecutor(12) as ex:
        return list(ex.map(job, [(i, variant, scale, h) for i in tr.image_id]))


fm = load_fm_encoder()
print("Mammo-FM encoder loaded, params:", sum(p.numel() for p in fm.parameters()) // 10**6, "M")
imnet = timm.create_model("tf_efficientnet_b5.ns_jft_in1k", pretrained=True, num_classes=0).eval()
M, S = torch.tensor([.485, .456, .406])[:, None, None], torch.tensor([.229, .224, .225])[:, None, None]


@torch.no_grad()
def imnet_features(imgs, bs=4):
    out = []
    for k in range(0, len(imgs), bs):
        x = torch.from_numpy(np.stack(imgs[k:k + bs]).astype(np.float32) / 255.)[:, None].repeat(1, 3, 1, 1)
        out.append(imnet((x - M) / S).numpy())
    return np.concatenate(out)


RUNS = [  # (backbone, variant, draft scale, harmonize). Order = priority.
    ("mammo_fm", "fm_protocol", 2, 0), ("mammo_fm", "bg_only", 2, 0), ("mammo_fm", "v2_bbox", 2, 0),
    ("mammo_fm", "bg_notext", 2, 0),    # background texture only (text inpainted) -> JPEG/noise leak test
    ("mammo_fm", "silhouette", 2, 0),   # breast shape only
    ("mammo_fm", "bg_notext", 2, 1),    # same after JPEG harmonisation
    ("mammo_fm", "fm_notext", 2, 1),    # FM protocol minus text
    ("mammo_fm", "fm_flip", 2, 1),      # + orientation normalisation
    ("mammo_fm", "fm_flatbg", 2, 1),    # FM geometry, text + bg texture removed (leak-free candidate)
    ("mammo_fm", "hp_tissue", 2, 1),    # tissue noise texture only: leak inside breast at 1/2 decode
    ("mammo_fm", "fm_flatbg", 4, 1),    # resolution: 1/4 decode upsampled
    ("mammo_fm", "hp_tissue", 4, 1),    # tissue leak at 1/4 decode
    ("imagenet_b5", "fm_flatbg", 2, 1), # same prep, ImageNet weights -> value of domain pretraining
]
rows, samples = [], {}
for bb, v, sc, h in RUNS:
    f = OUT / f"feat_{bb}_{v}_s{sc}{'_h' if h else ''}.npy"
    if not f.exists():
        t0 = time.time(); im = images(v, sc, h); samples[(v, sc)] = im[:3]
        np.save(f, fm_features(fm, im) if bb == "mammo_fm" else imnet_features(im))
        print(f"  extracted {f.name} in {time.time() - t0:.0f}s", flush=True)
    r = dict(backbone=bb, variant=v, draft_scale=sc, harmonize=h, **probe(np.load(f))); rows.append(r); print(r, flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT / "probe_results.csv", index=False)
print("\n", res.to_string(index=False))

# visual: one sample per class for each variant at scale 2
pk = [tr[tr.label == c].sample(1, random_state=7).image_id.item() for c in C]
vs = ["fm_protocol", "fm_notext", "fm_flip", "v2_bbox", "v2_bbox_pct", "bg_notext", "silhouette"]
fig, ax = plt.subplots(3, len(vs), figsize=(2.6 * len(vs), 13))
for r, i in enumerate(pk):
    for c, v in enumerate(vs):
        ax[r, c].imshow(fm_variant(img_path(i), v, 2), cmap="gray"); ax[r, c].set_title(f"{C[r]} {i[6:10]}\n{v}", fontsize=9); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT / "fm_variants.png", dpi=55)
