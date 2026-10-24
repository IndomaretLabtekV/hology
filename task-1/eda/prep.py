"""Shared preprocessing: grayscale -> trim border -> breast mask (largest CC) -> drop text -> flip to left -> crop -> resize/pad.
Used by the EDA scripts, the cache builder and the model (train/test parity)."""
from pathlib import Path
import numpy as np, cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data"
OUT_H, OUT_W = 1024, 768  # keeps the native 4:3 aspect (4740x3540)


def load_gray(p, scale=4):
    """Downscaled 8-bit grayscale. JPEG draft = DCT-domain downscale, identical for every file (normalizes q-table differences)."""
    im = Image.open(p)
    im.draft("L", (im.width // scale, im.height // scale))
    return np.asarray(im.convert("L"))


def breast_mask(a, thr=None):
    """Largest connected component of the thresholded image = breast. Text labels / markers are separate small blobs."""
    b = cv2.GaussianBlur(a, (5, 5), 0)
    # background is usually ~0 but some exports have a gray (~50) background -> threshold relative to bg level
    otsu = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    bg = int(np.argmax(np.bincount(b.ravel(), minlength=256)[:max(1, int(otsu))]))  # dark peak below Otsu
    t = thr if thr is not None else bg + max(15, 0.5 * (otsu - bg))
    m = (b > t).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n < 2:
        return np.ones_like(a, bool)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    mk = (lab == k).astype(np.uint8)
    # fill holes (dark fatty regions inside breast)
    cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(mk, cnts, -1, 1, cv2.FILLED)
    return mk.astype(bool)


def trim_border(a, frac=0.01):
    """Scanner leaves bright strips on the image edges; cut a thin frame."""
    h, w = a.shape
    dy, dx = int(h * frac), int(w * frac)
    return a[dy:h - dy, dx:w - dx]


def preprocess_v2(p, clahe=False, out=(OUT_H, OUT_W), dilate=31):
    """Chosen after EDA 07: tight crop+zoom LOST signal (skin line / periphery / physical scale).
    v2 = trim border -> flip chest wall left -> zero outside a *dilated* breast mask (text gone) -> resize full canvas."""
    a = trim_border(load_gray(p))
    m = breast_mask(a)
    if m[:, : m.shape[1] // 2].sum() < m[:, m.shape[1] // 2:].sum():
        a, m = a[:, ::-1], m[:, ::-1]
    mk = cv2.dilate(m.astype(np.uint8), np.ones((dilate, dilate), np.uint8)).astype(bool)
    a = np.where(mk, a, 0).astype(np.uint8)
    if clahe:
        a = np.where(mk, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(a), 0).astype(np.uint8)
    return cv2.resize(np.ascontiguousarray(a), out[::-1], interpolation=cv2.INTER_AREA)


def preprocess(p, clahe=True, out=(OUT_H, OUT_W), return_steps=False):
    a0 = load_gray(p)
    a = trim_border(a0)
    m = breast_mask(a)
    # flip so the chest wall is on the left (breast mass more on left half)
    flipped = m[:, : m.shape[1] // 2].sum() < m[:, m.shape[1] // 2:].sum()
    if flipped:
        a, m = a[:, ::-1], m[:, ::-1]
    am = np.where(m, a, 0).astype(np.uint8)  # kills text labels + background noise
    ys, xs = np.where(m)
    crop = am[ys.min():ys.max() + 1, 0:xs.max() + 1]  # keep chest-wall edge at x=0
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop)
        c[crop == 0] = 0
    else:
        c = crop
    # resize keeping aspect, pad right/bottom with 0
    H, W = out
    s = min(H / c.shape[0], W / c.shape[1])
    r = cv2.resize(c, (max(1, int(c.shape[1] * s)), max(1, int(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H, W), np.uint8)
    y0 = (H - r.shape[0]) // 2
    canvas[y0:y0 + r.shape[0], :r.shape[1]] = r
    if return_steps:
        return canvas, dict(raw=a0, trimmed=a, mask=m, masked=am, crop=crop, clahe=c, flipped=flipped)
    return canvas


def img_path(image_id):
    for s in ("train", "test"):
        p = D / f"{s}_images" / image_id
        if p.exists():
            return p
    raise FileNotFoundError(image_id)


# ---- augmentations validated in EDA 06/06b (operate on 1xHxW float tensors in [0,1]) ----
import torch


class ReflectRotate:
    """Rotation with mirrored border: no black wedge at the chest wall (RandomRotation creates one, EDA 06b)."""
    def __init__(self, deg=15, scale=(1.0, 1.0)): self.deg, self.scale = deg, scale
    def __call__(self, x):
        a = x[0].numpy(); h, w = a.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(np.random.uniform(-self.deg, self.deg)), float(np.random.uniform(*self.scale)))
        return torch.from_numpy(cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101))[None]


class MaskedGamma:
    """Gamma + contrast only inside tissue: background stays 0 (ColorJitter lifts it to gray, EDA 06)."""
    def __init__(self, g=(0.7, 1.4), c=(0.85, 1.15)): self.g, self.c = g, c
    def __call__(self, x):
        fg = x > 0
        y = x.clamp(1e-4, 1) ** float(np.random.uniform(*self.g)) * float(np.random.uniform(*self.c))
        return torch.where(fg, y.clamp(0, 1), x)
