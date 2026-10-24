"""Shared loaders for the EDA scripts. Caches all tiles once as uint8 .npy for speed."""
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "eda"
CACHE = ROOT / "outputs" / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

TRAIN_IMG = sorted((DATA / "train" / "images").glob("*.png"))
TRAIN_MSK = sorted((DATA / "train" / "masks").glob("*.png"))
TEST_IMG = sorted((DATA / "test_images").glob("*.png"))


def _stack(files, name, gray=False):
    p = CACHE / f"{name}.npy"
    if not p.exists():
        arr = np.stack([np.array(Image.open(f).convert("L" if gray else "RGB")) for f in files])
        np.save(p, arr)
    return np.load(p, mmap_mode="r")


def train_images():
    return _stack(TRAIN_IMG, "train_img")


def train_masks():
    return (_stack(TRAIN_MSK, "train_msk", gray=True) > 127).astype(np.uint8)


def test_images():
    return _stack(TEST_IMG, "test_img")


def out(name):
    return OUT / name


def nodata_mask(img, min_area=4096):
    """True NoData = large connected pure-black region (AOI clipping), not dark shadow/water pixels."""
    from scipy import ndimage as ndi
    black = img.max(-1) == 0
    lab, n = ndi.label(black)
    if n == 0:
        return black
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_area
    keep[0] = False
    return keep[lab]


def patch_feats(img, P=128):
    """Per 128-px fully-valid patch: position, colour, saturation, texture/sharpness, saturation-clip stats (used by 11/12/14)."""
    import cv2
    a = np.ascontiguousarray(img)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV).astype(np.float32)
    gm = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    lap_b = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 2), cv2.CV_32F))
    v = a.max(-1) > 0; f = []
    for y in range(0, 1024, P):
        for x in range(0, 1024, P):
            s = (slice(y, y + P), slice(x, x + P))
            if v[s].mean() < 0.95: continue
            px = a[s].reshape(-1, 3).astype(np.float32)
            f.append([y, x, *px.mean(0), *px.std(0), hsv[s][..., 1].mean(), hsv[s][..., 1].std(), hsv[s][..., 2].mean(),
                      gm[s].mean(), lap[s].mean(), lap_b[s].mean(), lap[s].mean() / (lap_b[s].mean() + 1e-3),
                      (px == 255).any(1).mean(), (px.max(1) < 20).mean()])
    return f


FEAT = ["mR", "mG", "mB", "sR", "sG", "sB", "satm", "sats", "valm", "grad", "lap", "lap_s2", "hf_ratio", "sat255", "dark"]
