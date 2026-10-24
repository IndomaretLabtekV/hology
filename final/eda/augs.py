"""Data-driven augmentations derived from the EDA (numpy/cv2 only, HWC uint8 in -> uint8 out).
Every photometric op leaves the mask untouched (labels are polygons drawn through clouds/seams/stretch),
NoData pixels (large pure-black regions) stay exactly 0 so the model keeps seeing true NoData as 0."""
import numpy as np, pandas as pd, cv2
from pathlib import Path

_BANK = Path(__file__).resolve().parents[1] / "outputs" / "eda" / "04_overlap_photometric.csv"


def load_gain_bank(path=_BANK):
    """(N,2,3) array of per-channel (gain, offset) observed between overlapping tiles, both directions."""
    d = pd.read_csv(path)
    g = d[["gain_R", "gain_G", "gain_B"]].values; o = d[["off_R", "off_G", "off_B"]].values
    fwd = np.stack([g, o], 1); inv = np.stack([1 / g, -o / g], 1)
    return np.concatenate([fwd, inv]).astype(np.float32)


def valid_of(img):
    return img.max(-1) > 0


def apply_affine(img, gain, off, region=None):
    """Per-channel affine + clip (= what the per-tile stretch does to the same ground)."""
    v = valid_of(img) if region is None else (valid_of(img) & region)
    out = img.astype(np.float32)
    out[v] = np.clip(out[v] * gain + off, 0, 255)
    return out.round().astype(np.uint8)


def bank_affine(img, bank, rng):
    g, o = bank[rng.integers(len(bank))]
    return apply_affine(img, g, o)


def restretch(img, lo=1.0, hi=98.0):
    """Re-run the per-tile percentile stretch on this (crop of a) tile."""
    v = valid_of(img)
    if v.sum() < 1000: return img
    out = img.astype(np.float32)
    for c in range(3):
        a, b = np.percentile(out[..., c][v], [lo, hi])
        out[..., c][v] = np.clip((out[..., c][v] - a) / max(b - a, 1) * 255, 0, 255)
    return out.round().astype(np.uint8)


def seam_affine(img, bank, rng):
    """A second acquisition on one side of a straight row/col seam (observed in 15% train / 26% test tiles)."""
    h, w = img.shape[:2]
    k = int(rng.integers(h // 8, 7 * h // 8))
    region = np.zeros((h, w), bool)
    side = rng.integers(2)
    if rng.random() < .5: region[:k] = True
    else: region[:, :k] = True
    if side: region = ~region
    g, o = bank[rng.integers(len(bank))]
    return apply_affine(img, g, o, region)


def fractal_noise(h, w, rng, octaves=5):
    n = np.zeros((h, w), np.float32)
    for k in range(octaves):
        s = max(2, 2 ** (k + 2))
        layer = cv2.resize(rng.random((s, s)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        n += layer / (1.6 ** k)
    n -= n.min(); return n / (n.max() + 1e-6)


def clouds(img, rng, cover=(0.05, 0.35), opacity=(0.5, 0.95)):
    """Thin/puffy cumulus over the scene: soft fractal alpha blended toward bright grey-white."""
    h, w = img.shape[:2]
    n = fractal_noise(h, w, rng)
    t = np.quantile(n, 1 - rng.uniform(*cover))
    alpha = np.clip((n - t) / (1 - t + 1e-6) * 2.5, 0, 1) * rng.uniform(*opacity)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 6)[..., None]
    white = np.array(rng.uniform(215, 250, 3), np.float32)
    out = img.astype(np.float32) * (1 - alpha) + white * alpha
    out[~valid_of(img)] = 0
    return out.round().astype(np.uint8)


def haze(img, rng, strength=(0.1, 0.35)):
    """Uniform bluish haze (the 'blue-cast' acquisitions): lift + desaturate."""
    s = rng.uniform(*strength); tint = np.array([180, 190, 225], np.float32)
    out = img.astype(np.float32) * (1 - s) + tint * s
    out[~valid_of(img)] = 0
    return out.round().astype(np.uint8)


def blur(img, rng, sigma=(0.4, 1.2)):
    out = cv2.GaussianBlur(img, (0, 0), rng.uniform(*sigma))
    out[~valid_of(img)] = 0
    return out


def hsv_jitter(img, rng, hue=12, sat=0.3):
    """Generic ColorJitter-style hue/sat jitter (NOT supported by the EDA: per-channel gain spread is tiny)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-hue, hue)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * rng.uniform(1 - sat, 1 + sat), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB); out[~valid_of(img)] = 0
    return out


def d4(img, msk, rng):
    k = int(rng.integers(4)); img, msk = np.rot90(img, k), np.rot90(msk, k)
    if rng.random() < .5: img, msk = img[:, ::-1], msk[:, ::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(msk)


def rotate_scale(img, msk, rng, max_deg=180, scale=(0.8, 1.25)):
    """Arbitrary rotation + mild scale. Corners fill with 0 = NoData, mask 0 there (consistent with NoData rule)."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-max_deg, max_deg), rng.uniform(*scale))
    im = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return im, mk


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    im = (rng.random((64, 64, 3)) * 255).astype(np.uint8); im[:10] = 0; mk = (rng.random((64, 64)) > .5).astype(np.uint8)
    bank = load_gain_bank()
    for f in (lambda x: bank_affine(x, bank, rng), lambda x: seam_affine(x, bank, rng), lambda x: clouds(x, rng),
              lambda x: haze(x, rng), lambda x: blur(x, rng), lambda x: hsv_jitter(x, rng), restretch):
        o = f(im); assert o.dtype == np.uint8 and o.shape == im.shape and (o[:10] == 0).all(), f
    a, b = d4(im, mk, rng); assert a.shape == im.shape and set(np.unique(b)) <= {0, 1}
    a, b = rotate_scale(im, mk, rng); assert set(np.unique(b)) <= {0, 1}
    print("augs self-check ok, bank size", len(bank))
