"""EDA 13: view + laterality from the burned-in text (OCR), then constrained patient pairing.
(A shape-based CC/MLO heuristic was tried first and failed: 3/4 known CC images read as MLO, CC/MLO counts unbalanced.)
Facts: each case = CC + MLO of ONE breast (confirmed pairs 0101/0209, 0010/0071, 0017/0181, 0138/0185);
split is patient-disjoint -> pairs never cross train/test. Pairing = bipartite matching CC<->MLO inside each
(split, laterality) cell. Validation: #CC == #MLO per cell; train pairs must share labels (random ~0.34)."""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from rapidocr_onnxruntime import RapidOCR
sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT, img_path, load_gray

OUT = ROOT / "outputs/13"; OUT.mkdir(parents=True, exist_ok=True)
g = pd.read_csv(ROOT / "outputs/05/groups.csv")[["image_id", "label"]]
pr = pd.read_csv(ROOT / "outputs/03/prep_stats.csv")[["image_id", "split", "flipped"]]
df = g.merge(pr, on="image_id")
KNOWN = [("0101", "0209"), ("0010", "0071"), ("0017", "0181"), ("0138", "0185")]

ocr = RapidOCR()
def read_tags(i):
    r, _ = ocr(load_gray(img_path(i), 4))
    txt = " ".join(t for _, t, s in (r or []) if float(s) > 0.5).upper().replace("0", "O")
    lat = re.search(r"\b([LR])\s*(?:CC|MLO|ML|LM|XCCL|CV)", txt)
    view = re.search(r"(MLO|XCCL|CC|ML|LM)", txt.replace("AX", " "))
    return dict(image_id=i, ocr=txt, lat=lat.group(1) if lat else None, view=view.group(1) if view else None, ax="AX" in txt)

T = pd.DataFrame([read_tags(i) for i in df.image_id]); df = df.merge(T, on="image_id")
df.to_csv(OUT / "ocr_tags.csv", index=False)
print("== raw OCR view / laterality counts ==\n", df.view.value_counts(dropna=False), "\n", df.lat.value_counts(dropna=False))
print("\nunreadable:\n", df[df.view.isna() | df.lat.isna()][["image_id", "split", "label", "ocr"]])
df["view2"] = df.view.map({"CC": "CC", "XCCL": "CC", "MLO": "MLO", "ML": "MLO", "LM": "MLO"})
print("\n== laterality vs flipped flag (is flip a laterality proxy?) ==\n", pd.crosstab(df.lat, df.flipped))
print("\n== balance: #CC vs #MLO per (split, laterality) -> equal if each case = CC+MLO of one breast ==")
print(pd.crosstab([df.split, df.lat], df.view2))
print("\n== 'AX' marker vs view ==\n", pd.crosstab(df.view2, df.ax))
print("\n== view by label (train) ==\n", pd.crosstab(df.label, df.view2))
for a, b in KNOWN:
    d = df[df.image_id.isin([f"image_{a}.jpg", f"image_{b}.jpg"])]
    print(f"known pair {a}+{b}: {d.lat.tolist()} {d.view2.tolist()}")

fig, ax = plt.subplots(2, 10, figsize=(22, 6))
for r, v in enumerate(["CC", "MLO"]):
    for c, row in enumerate(df[df.view2 == v].sample(10, random_state=1).itertuples()):
        ax[r, c].imshow(load_gray(img_path(row.image_id), 8), cmap="gray"); ax[r, c].set_title(f"{row.lat} {v} {row.image_id[6:10]}", fontsize=8); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(OUT / "ocr_views.png", dpi=55)

# ---------------- pairing ----------------
# laterality: OCR (display flip is sometimes mirrored by the operator), fallback to flip flag for the 3 text-less images
df["lat2"] = df.lat.fillna(df.flipped.map({True: "R", False: "L"}))
# text-less images get the view that balances their (split, lat) cell
for i in df.index[df.view2.isna()]:
    cell = df[(df.split == df.split[i]) & (df.lat2 == df.lat2[i])]
    df.loc[i, "view2"] = "MLO" if (cell.view2 == "CC").sum() > (cell.view2 == "MLO").sum() else "CC"
print("\n== balance after filling text-less images ==\n", pd.crosstab([df.split, df.lat2], df.view2))

ids = pd.read_csv(ROOT / "outputs/05/groups.csv").image_id.tolist()
E = np.load(ROOT / "outputs/05/emb_convnext.npy"); E = pd.DataFrame(E, index=ids).loc[df.image_id].values
En = E / np.linalg.norm(E, axis=1, keepdims=True)
def hist(i):
    a = load_gray(img_path(i), 8); v = a[a > 20]; h = np.histogram(v, 64, (0, 256))[0].astype(float); return np.sqrt(h / h.sum())
H = np.stack([hist(i) for i in df.image_id])
tf = pd.read_csv(ROOT / "outputs/08/text_feats.csv").set_index("image_id").loc[df.image_id]
area = pd.read_csv(ROOT / "outputs/03/prep_stats.csv").set_index("image_id").loc[df.image_id]
fmt = df.ocr.fillna("").str.contains(r"[LR]\s+(CC|MLO|ML|LM)").values  # "L CC" vs "LCC": export-software format
SIG = {
    "cnn": En @ En.T,
    "hist": H @ H.T,
    "text_style": -np.abs(tf.char_h_med.values[:, None] - tf.char_h_med.values[None]) / 10 + (fmt[:, None] == fmt[None]),
    "breast_size": -np.abs(np.log(area.mask_frac.values[:, None]) - np.log(area.mask_frac.values[None])),
}
z = lambda M: (M - M.mean()) / M.std()
SIG["cnn+hist"] = z(SIG["cnn"]) + z(SIG["hist"])
SIG["all"] = z(SIG["cnn"]) + z(SIG["hist"]) + z(SIG["text_style"]) + z(SIG["breast_size"])


def match(S):
    pairs = []
    for (sp, lt), d in df.groupby(["split", "lat2"]):
        cc, ml = d.index[d.view2 == "CC"].values, d.index[d.view2 == "MLO"].values
        r, c = linear_sum_assignment(-S[np.ix_(cc, ml)])
        pairs += [(cc[a], ml[b]) for a, b in zip(r, c)]
    return pairs

lab = df.label.values
print("\n== pairing quality (train): same-label rate of matched pairs, 1.0 = perfect, random ~0.34 ==")
res = {}
for k, S in SIG.items():
    pr = match(S); tt = [(a, b) for a, b in pr if lab[a] != "?"]
    res[k] = np.mean([lab[a] == lab[b] for a, b in tt]); print(f"{k:12s} pairs={len(pr)}  same-label={res[k]:.3f}")
best = max(res, key=res.get); pairs = match(SIG[best])
df["pair"] = -1
for n, (a, b) in enumerate(pairs): df.loc[[a, b], "pair"] = n
df["pair"] = np.where(df.pair < 0, 1000 + df.index, df.pair)
df.to_csv(OUT / "pairs.csv", index=False)
print(f"using '{best}'. known pairs recovered:",
      [df.set_index('image_id').pair[f'image_{a}.jpg'] == df.set_index('image_id').pair[f'image_{b}.jpg'] for a, b in KNOWN])

fig, ax = plt.subplots(4, 8, figsize=(16, 14))
for k, (a, b) in enumerate(pairs[:16]):
    for t, q in enumerate((a, b)):
        x = ax.flat[2 * k + t]; x.imshow(load_gray(img_path(df.image_id[q]), 8), cmap="gray"); x.axis("off")
        x.set_title(f"{df.image_id[q][6:10]} {df.lat2[q]}{df.view2[q]} {lab[q]}", fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "pairs.png", dpi=55)
