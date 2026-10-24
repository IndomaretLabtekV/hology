"""EDA 09: build cache_v2 (and cache_v2c with CLAHE) with preprocess_v2 + save 3 samples/class for visual check."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, str(Path(__file__).parent))
from prep import preprocess_v2, img_path, ROOT

g = pd.read_csv(ROOT / "outputs/05/groups.csv")
OUT = ROOT / "outputs/09"; OUT.mkdir(parents=True, exist_ok=True)
for name, cl in [("cache_v2", False), ("cache_v2c", True)]:
    (ROOT / name).mkdir(exist_ok=True)

def job(i):
    for name, cl in [("cache_v2", False), ("cache_v2c", True)]:
        cv2.imwrite(str(ROOT / name / i.replace(".jpg", ".png")), preprocess_v2(img_path(i), clahe=cl))

with ProcessPoolExecutor(16) as ex: list(ex.map(job, g.image_id))
pk = pd.concat([g[g.label == l].sample(3, random_state=5) for l in ["Normal", "Benign", "Malignant", "?"]])
fig, ax = plt.subplots(2, len(pk), figsize=(2 * len(pk), 6))
for j, r in enumerate(pk.itertuples()):
    for i, name in enumerate(["cache_v2", "cache_v2c"]):
        ax[i, j].imshow(cv2.imread(str(ROOT / name / r.image_id.replace(".jpg", ".png")), 0), cmap="gray")
        ax[i, j].set_title(f"{r.label} {r.image_id[6:10]}\n{name}", fontsize=7); ax[i, j].axis("off")
plt.tight_layout(); plt.savefig(OUT / "v2_samples.png", dpi=60)
print("done")
