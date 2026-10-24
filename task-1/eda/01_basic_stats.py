"""EDA 01: label distribution, resolution, mode, file size, intensity stats (train vs test)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from PIL import Image

D = Path(__file__).resolve().parents[1] / "data"
OUT = Path(__file__).resolve().parents[1] / "outputs/01"; OUT.mkdir(parents=True, exist_ok=True)

tr = pd.read_csv(D / "train.csv"); tr["split"] = "train"
te = pd.read_csv(D / "test.csv"); te["split"] = "test"; te["label"] = "?"
df = pd.concat([tr, te], ignore_index=True)

rows = []
for r in df.itertuples():
    p = D / f"{r.split}_images" / r.image_id
    im = Image.open(p)
    a = np.asarray(im.convert("L"))
    rgb = np.asarray(im.convert("RGB")).astype(np.int16)
    rows.append(dict(image_id=r.image_id, W=im.width, H=im.height, mode=im.mode,
                     kb=p.stat().st_size / 1024, mean=a.mean(), std=a.std(),
                     frac_black=(a < 10).mean(), frac_white=(a > 245).mean(),
                     color_diff=np.abs(rgb[..., 0] - rgb[..., 2]).mean(),  # is it really grayscale?
                     dpi=str(im.info.get("dpi")), idnum=int(r.image_id[6:10])))
m = df.merge(pd.DataFrame(rows), on="image_id")
m["aspect"] = m.H / m.W
m.to_csv(OUT / "stats.csv", index=False)

pd.set_option("display.width", 200)
print("== label counts ==\n", tr.label.value_counts(), "\n", tr.label.value_counts(normalize=True).round(3))
print("\n== modes ==\n", m.groupby("split")["mode"].value_counts())
print("\n== unique resolutions (W x H) ==\n", m.groupby(["split", "W", "H"]).size().sort_values(ascending=False).head(20))
print("\n== numeric stats by label ==")
print(m.groupby("label")[["W", "H", "aspect", "kb", "mean", "std", "frac_black", "frac_white", "color_diff"]].describe().T.round(3).to_string())
print("\n== dpi ==\n", m.dpi.value_counts().head())
print("\n== image_id number vs label (is ID ordering informative?) ==")
print(m[m.split == "train"].groupby("label").idnum.describe().round(1))
# id ordering: runs of consecutive labels
t = m[m.split == "train"].sort_values("idnum")
runs = (t.label != t.label.shift()).cumsum()
print("n label runs in id order:", runs.nunique(), "of", len(t), "(random would be ~", round(len(t) * (1 - (t.label.value_counts(normalize=True) ** 2).sum())), ")")

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
tr.label.value_counts().plot.bar(ax=ax[0, 0], title="label counts")
for lab, g in m.groupby("label"):
    ax[0, 1].scatter(g.W, g.H, label=lab, alpha=.6, s=15)
    ax[0, 2].hist(g["mean"], bins=30, alpha=.5, label=lab)
    ax[1, 0].hist(g.kb, bins=30, alpha=.5, label=lab)
    ax[1, 1].hist(g.frac_black, bins=30, alpha=.5, label=lab)
    ax[1, 2].scatter(g.idnum, g["mean"], label=lab, s=10)
for a, t_ in zip(ax.flat[1:], ["W vs H", "mean intensity", "file KB", "frac black (<10)", "id number vs mean"]):
    a.set_title(t_); a.legend()
plt.tight_layout(); plt.savefig(OUT / "basic_stats.png", dpi=90)
print("saved", OUT)
