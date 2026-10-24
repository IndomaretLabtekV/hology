"""EDA 05: exact/near duplicates + pseudo-patient grouping.
Dataset paper: 133 cases x 2 images. Same patient -> same label. We find likely pairs with
(1) perceptual hash on raw thumbnails (exact/near dup), (2) ImageNet CNN embeddings of the preprocessed cache,
(3) raw global intensity histogram (same exposure/windowing session). Output: groups.csv used for GroupKFold."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, torch, timm, matplotlib.pyplot as plt, imagehash
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from prep import img_path, load_gray, trim_border, ROOT

OUT = ROOT / "outputs/05"; OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(ROOT / "outputs/03/prep_stats.csv")
ids = m.image_id.tolist(); lab = m.label.values; n = len(ids)

# 1) phash on raw (unflipped) + flipped -> catches re-saved / mirrored copies
ph = [imagehash.phash(Image.fromarray(trim_border(load_gray(img_path(i), 8)))) for i in ids]
phf = [imagehash.phash(Image.fromarray(trim_border(load_gray(img_path(i), 8))[:, ::-1].copy())) for i in ids]
H = np.array([[min(ph[i] - ph[j], ph[i] - phf[j]) for j in range(n)] for i in range(n)]); np.fill_diagonal(H, 99)
print("min phash distance per pair (<=10 = near-dup):", np.sort(H[np.triu_indices(n, 1)])[:15])
dups = [(ids[i], ids[j], H[i, j], lab[i], lab[j]) for i, j in zip(*np.where(np.triu(H <= 10, 1)))]
print("near duplicates:", dups)

# 2) CNN embeddings on the preprocessed cache
net = timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=0).eval()
def emb(i):
    a = cv2.imread(str(ROOT / "cache" / i.replace(".jpg", ".png")), 0)
    x = torch.tensor(cv2.resize(a, (384, 512)) / 255., dtype=torch.float32)[None, None].repeat(1, 3, 1, 1)
    x = (x - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    return net(x)[0].numpy()
with torch.no_grad():
    E = np.stack([emb(i) for i in ids])
np.save(OUT / "emb_convnext.npy", E)
En = E / np.linalg.norm(E, axis=1, keepdims=True); S = En @ En.T; np.fill_diagonal(S, -1)

# 3) raw histogram of tissue (acquisition fingerprint: kVp/windowing of one session is shared by both views)
def hist(i):
    a = trim_border(load_gray(img_path(i), 8)); v = a[a > 20]
    h = np.histogram(v, 64, (0, 256))[0].astype(float); return np.sqrt(h / h.sum())
Hh = np.stack([hist(i) for i in ids]); Sh = Hh @ Hh.T; np.fill_diagonal(Sh, -1)

prior_same = sum((lab == l).sum() * ((lab == l).sum() - 1) for l in set(lab) if l != "?") / ((lab != "?").sum() * ((lab != "?").sum() - 1))
for name, M in [("cnn", S), ("hist", Sh), ("cnn+hist", S / S.std() + Sh / Sh.std())]:
    nn = M.argmax(1)
    mutual = [(i, nn[i]) for i in range(n) if nn[nn[i]] == i and i < nn[i]]
    both_tr = [(i, j) for i, j in mutual if lab[i] != "?" and lab[j] != "?"]
    same = np.mean([lab[i] == lab[j] for i, j in both_tr])
    cross = sum((lab[i] == "?") != (lab[j] == "?") for i, j in mutual)
    print(f"[{name}] mutual-NN pairs={len(mutual)}  same-label rate (train-train)={same:.2f} vs chance {prior_same:.2f}  train<->test pairs={cross}")

# visual check of the top-24 cnn+hist mutual pairs
M = S / S.std() + Sh / Sh.std(); nn = M.argmax(1)
mutual = sorted([(M[i, nn[i]], i, nn[i]) for i in range(n) if nn[nn[i]] == i and i < nn[i]], reverse=True)
fig, ax = plt.subplots(6, 8, figsize=(16, 22))
for k, (s, i, j) in enumerate(mutual[:24]):
    for t, q in enumerate((i, j)):
        a = ax.flat[2 * k + t]; a.axis("off")
        a.imshow(load_gray(img_path(ids[q]), 8), cmap="gray"); a.set_title(f"#{k} {ids[q][6:10]} {lab[q]}\ns={s:.1f}", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "top_pairs.png", dpi=60)
pd.DataFrame([(ids[i], ids[j], s, lab[i], lab[j]) for s, i, j in mutual], columns=["a", "b", "score", "la", "lb"]).to_csv(OUT / "mutual_pairs.csv", index=False)

# 4) constrained pairing: both views of one case share laterality -> same `flipped` flag. Greedy max-score matching.
fl = m.flipped.values
Mc = np.where(fl[:, None] == fl[None, :], M, -np.inf)
nn = Mc.argmax(1)
mutual = [(i, nn[i]) for i in range(n) if nn[nn[i]] == i and i < nn[i]]
tt = [(i, j) for i, j in mutual if lab[i] != "?" and lab[j] != "?"]
print(f"[cnn+hist+laterality] mutual pairs={len(mutual)} same-label={np.mean([lab[i] == lab[j] for i, j in tt]):.2f} "
      f"train<->test={sum((lab[i] == '?') != (lab[j] == '?') for i, j in mutual)} (true pairs should give 0)")
# groups: union of mutual pairs; everything else singleton. ponytail: pairing is heuristic, over-grouping is the safe direction
g = np.arange(n)
for i, j in mutual: g[j] = g[i]
pd.DataFrame({"image_id": ids, "label": lab, "group": g}).to_csv(ROOT / "outputs/05/groups.csv", index=False)
print("groups:", len(set(g)), "for", n, "images")
