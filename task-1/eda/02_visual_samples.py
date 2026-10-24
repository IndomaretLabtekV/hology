"""EDA 02: visual grid of samples per class + outlier-resolution images + JPEG metadata by class."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; D = ROOT / "data"
OUT = ROOT / "outputs/02"; OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(ROOT / "outputs/01/stats.csv")
path = lambda r: D / f"{r.split}_images" / r.image_id

# JPEG metadata: quantization table (quality proxy), subsampling, mode -> per class
def jpeg_meta(p):
    im = Image.open(p)
    q = im.quantization if hasattr(im, "quantization") else {}
    q0 = np.mean(q[0]) if q else np.nan
    return dict(q0_mean=q0, n_qtables=len(q), progressive=im.info.get("progressive", 0),
                exif=bool(im.info.get("exif")), icc=bool(im.info.get("icc_profile")), jfif=im.info.get("jfif_version"))
meta = pd.DataFrame([jpeg_meta(path(r)) | {"image_id": r.image_id} for r in m.itertuples()])
m = m.merge(meta, on="image_id")
m.to_csv(OUT / "stats_meta.csv", index=False)
print("== mode by label ==\n", pd.crosstab(m.label, m["mode"]))
print("\n== quant table mean (lower = higher quality) by label ==\n", m.groupby("label").q0_mean.describe().round(2))
print("\n== jfif / exif / icc / dpi by label ==")
for c in ["jfif", "exif", "icc", "dpi", "n_qtables", "progressive"]:
    print(pd.crosstab(m.label, m[c].astype(str)), "\n")
print("== odd resolution images ==\n", m[m.W != 3540][["image_id", "label", "W", "H", "mode", "kb"]])

rng = np.random.default_rng(0)
labs = ["Normal", "Benign", "Malignant", "?"]
fig, ax = plt.subplots(4, 8, figsize=(24, 17))
for i, lab in enumerate(labs):
    g = m[m.label == lab].sample(8, random_state=0)
    for j, r in enumerate(g.itertuples()):
        im = Image.open(path(r)).convert("L"); im.thumbnail((512, 512))
        ax[i, j].imshow(im, cmap="gray"); ax[i, j].set_title(f"{lab} {r.image_id[6:10]} {r.mode} {r.kb:.0f}KB", fontsize=9); ax[i, j].axis("off")
plt.tight_layout(); plt.savefig(OUT / "grid_per_class.png", dpi=70)

odd = m[m.W != 3540]
fig, ax = plt.subplots(1, len(odd), figsize=(8, 6))
for a, r in zip(np.atleast_1d(ax), odd.itertuples()):
    im = Image.open(path(r)).convert("L"); im.thumbnail((512, 512)); a.imshow(im, cmap="gray"); a.set_title(r.image_id)
plt.savefig(OUT / "odd_resolution.png", dpi=70)
print("saved", OUT)
