"""EDA 03: run preprocessing on 3 samples/class, save every step, then build the full cache (cache/*.png).
Also asserts the breast mask sanity (area fraction, text removed)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, cv2
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from prep import preprocess, img_path, ROOT

OUT = ROOT / "outputs/03"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = ROOT / "cache"; CACHE.mkdir(exist_ok=True)
m = pd.read_csv(ROOT / "outputs/01/stats.csv")

picks = pd.concat([m[m.label == l].sample(3, random_state=1) for l in ["Normal", "Benign", "Malignant"]] +
                  [m[m.W != 3540]])
steps = ["raw", "mask", "masked", "crop", "clahe", "final"]
fig, ax = plt.subplots(len(picks), len(steps), figsize=(3 * len(steps), 3.6 * len(picks)))
for i, r in enumerate(picks.itertuples()):
    fin, st = preprocess(img_path(r.image_id), return_steps=True)
    st["final"] = fin
    for j, s in enumerate(steps):
        ax[i, j].imshow(st[s], cmap="gray"); ax[i, j].axis("off")
        ax[i, j].set_title(f"{r.label} {r.image_id[6:10]} {s}" + (" (flipped)" if s == "final" and st["flipped"] else ""), fontsize=8)
plt.tight_layout(); plt.savefig(OUT / "steps.png", dpi=60)


def job(iid):
    fin, st = preprocess(img_path(iid), return_steps=True)
    cv2.imwrite(str(CACHE / iid.replace(".jpg", ".png")), fin)
    m_ = st["mask"]
    return dict(image_id=iid, mask_frac=m_.mean(), flipped=st["flipped"], crop_h=st["crop"].shape[0], crop_w=st["crop"].shape[1],
                touches_left=m_[:, 0].any())

with ProcessPoolExecutor(16) as ex:
    res = pd.DataFrame(list(ex.map(job, m.image_id)))
res = m[["image_id", "label", "split"]].merge(res)
res.to_csv(OUT / "prep_stats.csv", index=False)
print(res.groupby("label")[["mask_frac", "crop_h", "crop_w"]].describe().T.round(3).to_string())
print("\nflipped by label:\n", pd.crosstab(res.label, res.flipped))
sus = res[(res.mask_frac < 0.1) | (res.mask_frac > 0.8)]
print("\nsuspicious masks:\n", sus)

# contact sheet of ALL final images, sorted by mask_frac (eyeball failures)
ids = res.sort_values("mask_frac").image_id.tolist()
n = len(ids); cols = 19
fig, ax = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(cols * 1.2, np.ceil(n / cols) * 1.6))
for a in ax.flat: a.axis("off")
for a, iid in zip(ax.flat, ids):
    a.imshow(cv2.imread(str(CACHE / iid.replace(".jpg", ".png")), 0)[::4, ::4], cmap="gray"); a.set_title(iid[6:10], fontsize=6)
plt.tight_layout(); plt.savefig(OUT / "all_final.png", dpi=80)
print("saved", OUT)
