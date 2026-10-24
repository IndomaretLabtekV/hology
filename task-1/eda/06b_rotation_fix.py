"""EDA 06b: fix the black chest-wall wedge created by RandomRotation. Metric: wedge = rows where the original
touches the left edge with tissue but the augmented image is black there (artificial edge never seen at test)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2, matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT
OUT = ROOT / "outputs/06"; torch.manual_seed(0)
m = pd.read_csv(ROOT / "outputs/03/prep_stats.csv"); m = m[m.split == "train"]
load = lambda i: torch.tensor(cv2.resize(cv2.imread(str(ROOT / "cache" / i.replace(".jpg", ".png")), 0), (384, 512)) / 255., dtype=torch.float32)[None]
H, W = 512, 384
CANDS = {
    "rot15_center": T.RandomRotation(15, fill=0),
    "rot10_pivot_chestwall": T.RandomAffine(10, center=(0, H / 2), fill=0),
    "rot10_pivot+scale1.15": T.RandomAffine(10, scale=(1.1, 1.2), center=(0, H / 2), fill=0),
    "rot15_pivot+scale1.25": T.RandomAffine(15, scale=(1.2, 1.3), center=(0, H / 2), fill=0),
}
rows = []
for name, aug in CANDS.items():
    for r in m.sample(100, random_state=0).itertuples():
        x = load(r.image_id); left = x[0, :, :3].mean(1) > 0
        for _ in range(3):
            y = aug(x); yl = y[0, :, :3].mean(1) > 0.01
            fg0 = (x > 0).float().sum()
            rows.append(dict(aug=name, wedge=float((left & ~yl).float().sum() / max(1, left.sum())),
                             fg_lost=float(1 - ((y > 0).float().sum() / fg0).clamp(max=1)),
                             tip_cut=float((y[0, :, -2:] > 0).float().mean())))
print(pd.DataFrame(rows).groupby("aug").agg(['mean', 'max']).round(3).to_string())
pk = pd.concat([m[m.label == l].sample(1, random_state=3) for l in ["Normal", "Benign", "Malignant"]])
fig, ax = plt.subplots(3, 1 + 3 * 2, figsize=(20, 10))
for i, r in enumerate(pk.itertuples()):
    x = load(r.image_id); ax[i, 0].imshow(x[0], cmap="gray"); ax[i, 0].set_title(f"{r.label} orig")
    for k, name in enumerate(["rot15_center", "rot10_pivot+scale1.15"]):
        for j in range(3):
            a = ax[i, 1 + 3 * k + j]; a.imshow(CANDS[name](x)[0], cmap="gray"); a.set_title(name, fontsize=8)
for a in ax.flat: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "rotation_fix.png", dpi=55)


class ReflectRotate:
    """Rotate with mirrored border: the chest-wall wedge gets filled with reflected tissue instead of black."""
    def __init__(self, deg=15): self.deg = deg
    def __call__(self, x):
        a = x[0].numpy(); h, w = a.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(np.random.uniform(-self.deg, self.deg)), 1.0)
        return torch.from_numpy(cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))[None]

CANDS = {"rot15_center": T.RandomRotation(15, fill=0), "reflect_rot15": ReflectRotate(15), "reflect_rot25": ReflectRotate(25)}
rows = []
for name, aug in CANDS.items():
    for r in m.sample(100, random_state=0).itertuples():
        x = load(r.image_id); left = x[0, :, :3].mean(1) > 0
        for _ in range(3):
            y = aug(x); yl = y[0, :, :3].mean(1) > 0.01
            rows.append(dict(aug=name, wedge=float((left & ~yl).float().sum() / max(1, left.sum())),
                             bg_black_frac=float((y[0][x[0] == 0] < 0.02).float().mean())))
print(pd.DataFrame(rows).groupby("aug").agg(['mean', 'max']).round(3).to_string())
fig, ax = plt.subplots(3, 7, figsize=(20, 10))
for i, r in enumerate(pk.itertuples()):
    x = load(r.image_id); ax[i, 0].imshow(x[0], cmap="gray"); ax[i, 0].set_title(f"{r.label} orig")
    for j in range(6):
        nm = "rot15_center" if j < 3 else "reflect_rot15"
        ax[i, j + 1].imshow(CANDS[nm](x)[0], cmap="gray"); ax[i, j + 1].set_title(nm, fontsize=8)
for a in ax.flat: a.axis("off")
plt.tight_layout(); plt.savefig(OUT / "rotation_reflect.png", dpi=55)
