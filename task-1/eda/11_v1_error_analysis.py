"""EDA 11: where does baseline v1 (EfficientNet-B3, OOF macro F1 0.641) fail, and why?
Joins v1 OOF probabilities with image properties from earlier EDA and checks:
confidence of errors, error rate vs brightness / breast size / mode / text, pair consistency, calibration."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT, img_path, load_gray

OUT = ROOT / "outputs/11"; OUT.mkdir(parents=True, exist_ok=True)
C = ["Normal", "Benign", "Malignant"]
tr = pd.read_csv(ROOT / "data/train.csv")
P = np.load(ROOT / "results/v1/weights/oof.npy")
y = tr.label.map(C.index).values
tr["pred"] = P.argmax(1); tr["conf"] = P.max(1); tr["p_true"] = P[np.arange(len(y)), y]; tr["wrong"] = tr.pred != y
for k, c in enumerate(C): tr[f"p_{c}"] = P[:, k]

st = pd.read_csv(ROOT / "outputs/01/stats.csv")[["image_id", "mode", "kb", "mean", "frac_black"]]
pr = pd.read_csv(ROOT / "outputs/03/prep_stats.csv")[["image_id", "mask_frac", "flipped"]]
tx = pd.read_csv(ROOT / "outputs/08/text_feats.csv")[["image_id", "char_h_med", "text_area"]]
tr = tr.merge(st, on="image_id").merge(pr, on="image_id").merge(tx, on="image_id")
tr["tissue_mean"] = [load_gray(img_path(i), 8)[load_gray(img_path(i), 8) > 20].mean() for i in tr.image_id]

print(f"OOF macro F1 {f1_score(y, tr.pred, average='macro'):.4f}")
print("confusion (rows true N/B/M):\n", confusion_matrix(y, tr.pred))
print("\n== error rate by true class ==\n", tr.groupby("label").wrong.mean().round(3))
print("\n== confidence: correct vs wrong ==\n", tr.groupby("wrong").conf.describe().round(3))

# calibration: reliability by confidence bin
bins = pd.cut(tr.conf, [0, .4, .5, .6, .7, .8, 1.0])
print("\n== reliability (acc vs mean conf per bin) ==\n", tr.groupby(bins, observed=True).agg(n=("conf", "size"), conf=("conf", "mean"), acc=("wrong", lambda w: 1 - w.mean())).round(3))

# error vs properties (quartiles)
print("\n== error rate vs image properties (quartile bins) ==")
for f in ["tissue_mean", "mask_frac", "kb", "char_h_med"]:
    q = pd.qcut(tr[f], 4, duplicates="drop")
    print(f"{f}:", tr.groupby(q, observed=True).wrong.mean().round(2).to_dict())
print("flipped:", tr.groupby("flipped").wrong.mean().round(2).to_dict())
print("Normal saved as RGB (the 2 off-pattern normals):\n", tr[(tr.label == "Normal") & (tr["mode"] == "RGB")][["image_id", "pred", "conf"]])

# pair consistency with the pseudo-patient pairs from EDA 05
g = pd.read_csv(ROOT / "outputs/05/groups.csv")
tr = tr.merge(g[["image_id", "group"]], on="image_id")
multi = tr.groupby("group").filter(lambda d: len(d) == 2)
print(f"\n== pairs inside train: {len(multi) // 2} ==")
agree = multi.groupby("group").pred.nunique().eq(1).mean()
print(f"pred agreement inside pairs {agree:.2f}; label agreement {multi.groupby('group').label.nunique().eq(1).mean():.2f}")
Pp = P.copy()
for _, d in multi.groupby("group"):
    idx = d.index.values; Pp[idx] = P[idx].mean(0)
print(f"macro F1 if pair probabilities are averaged: {f1_score(y, Pp.argmax(1), average='macro'):.4f} (vs {f1_score(y, P.argmax(1), average='macro'):.4f})")

tr.to_csv(OUT / "v1_oof_joined.csv", index=False)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for w, d in tr.groupby("wrong"): ax[0].hist(d.conf, bins=20, alpha=.6, label=f"wrong={w}")
ax[0].legend(); ax[0].set_title("OOF confidence")
for c in C:
    d = tr[tr.label == c]; ax[1].hist(d.p_Malignant / (d.p_Malignant + d.p_Benign + 1e-9), bins=20, alpha=.5, label=c)
ax[1].set_title("P(Malig | abnormal)"); ax[1].legend()
ax[2].scatter(tr.tissue_mean, tr.p_true, c=tr.wrong, cmap="coolwarm", s=12); ax[2].set_xlabel("tissue mean (raw)"); ax[2].set_ylabel("p(true)")
plt.tight_layout(); plt.savefig(OUT / "error_stats.png", dpi=80)

# the most confident mistakes per true class
fig, ax = plt.subplots(3, 6, figsize=(18, 13))
for r, c in enumerate(C):
    d = tr[(tr.label == c) & tr.wrong].sort_values("conf", ascending=False).head(6)
    for k, row in enumerate(d.itertuples()):
        a = ax[r, k]; a.imshow(cv2.imread(str(ROOT / "cache_v2" / row.image_id.replace(".jpg", ".png")), 0), cmap="gray")
        a.set_title(f"{row.image_id[6:10]} true={c}\npred={C[row.pred]} conf={row.conf:.2f}", fontsize=9)
    for a in ax[r]: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "confident_errors.png", dpi=55)
