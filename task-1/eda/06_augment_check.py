"""EDA 06: augmentation sanity. For 2 images/class, show N random draws of each candidate augmentation.
Checks numerically: background stays black? breast cut off? intensity range collapsed?"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch, cv2, matplotlib.pyplot as plt
import torchvision.transforms.v2 as T
sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT

OUT = ROOT / "outputs/06"; OUT.mkdir(parents=True, exist_ok=True)
m = pd.read_csv(ROOT / "outputs/03/prep_stats.csv"); m = m[m.split == "train"]
torch.manual_seed(0)


class MaskedGamma:
    """Gamma/contrast only inside the breast: background stays exactly 0 (ColorJitter brightness lifts it)."""
    def __init__(self, g=(0.7, 1.4)): self.g = g
    def __call__(self, x):
        g = float(torch.empty(1).uniform_(*self.g)); fg = x > 0
        return torch.where(fg, x.clamp(1e-4, 1) ** g, x)


AUGS = {
    "rot15": T.RandomRotation(15, fill=0),
    "rot45 (paper)": T.RandomRotation(45, fill=0),
    "vflip": T.RandomVerticalFlip(1.0),
    "colorjitter": T.ColorJitter(brightness=0.3, contrast=0.3),
    "masked_gamma": MaskedGamma(),
    "rrc": T.RandomResizedCrop((512, 384), scale=(0.6, 1.0), ratio=(0.7, 0.8)),
    "affine": T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1), fill=0),
}

load = lambda i: torch.tensor(cv2.resize(cv2.imread(str(ROOT / "cache" / i.replace(".jpg", ".png")), 0), (384, 512)) / 255., dtype=torch.float32)[None]
picks = pd.concat([m[m.label == l].sample(2, random_state=3) for l in ["Normal", "Benign", "Malignant"]])

stats = []
for name, aug in AUGS.items():
    fig, ax = plt.subplots(len(picks), 5, figsize=(12, 3.3 * len(picks)))
    for r, row in enumerate(picks.itertuples()):
        x = load(row.image_id)
        ax[r, 0].imshow(x[0], cmap="gray", vmin=0, vmax=1); ax[r, 0].set_title(f"{row.label} {row.image_id[6:10]} orig", fontsize=8)
        for c in range(1, 5):
            y = aug(x)
            ax[r, c].imshow(y[0], cmap="gray", vmin=0, vmax=1); ax[r, c].set_title(name, fontsize=8)
    for a in ax.flat: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / f"aug_{name.split()[0]}.png", dpi=55); plt.close()
    # numeric: over 100 train images x 3 draws
    for row in m.sample(100, random_state=0).itertuples():
        x = load(row.image_id); fg0 = (x > 0).float().mean()
        for _ in range(3):
            y = aug(x)
            stats.append(dict(aug=name, bg_level=float(y[y < 0.02].mean()) if (y < 0.02).any() else np.nan,
                              bg_nonzero=float(((x == 0) & (y > 0.02)).float().mean() / max(1e-6, float((x == 0).float().mean()))),
                              fg_kept=float((y > 0).float().mean() / fg0),
                              chestwall_on_left=float((y[0, :, :8] > 0).float().mean()),
                              p99=float(y.quantile(.99))))
s = pd.DataFrame(stats)
print(s.groupby("aug").agg(['mean', 'min']).round(3).to_string())
