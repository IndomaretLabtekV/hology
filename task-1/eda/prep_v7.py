"""v7 additions on top of prep_v6 (shared verbatim with the v7 notebook): cover rotation, masked jitter, Mammo-FM view."""
import cv2
import numpy as np



def cover_scale(shape, angle):
    h, w = shape
    rad = np.deg2rad(abs(angle))
    return max((w * np.cos(rad) + h * np.sin(rad)) / w, (w * np.sin(rad) + h * np.cos(rad)) / h)


def rotate_cover(array, angle, nearest=False):
    # Zoom just enough that the rotated field covers the whole canvas: no black wedge, no mirrored tissue.
    h, w = array.shape
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, cover_scale(array.shape, angle))
    return cv2.warpAffine(array, m, (w, h), flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def masked_jitter(array, brightness, contrast):
    # Brightness/contrast only inside tissue, around the tissue mean; background stays exactly 0.
    tissue = array > 0
    if not tissue.any():
        return array.copy()
    f = array.astype(np.float32)
    mu = f[tissue].mean()
    out = np.clip((f - mu) * contrast + mu * brightness, 1, 255)
    return np.where(tissue, np.round(out), 0).astype(np.uint8)


def augment_v7(array, rng, degrees=7.0, brightness=(0.9, 1.1), contrast=(0.9, 1.1), hflip_p=0.0, vflip_p=0.5):
    a = rotate_cover(array, float(rng.uniform(-degrees, degrees)))
    a = masked_jitter(a, float(rng.uniform(*brightness)), float(rng.uniform(*contrast)))
    if rng.random() < vflip_p:
        a = a[::-1]
    if rng.random() < hflip_p:
        a = a[:, ::-1]
    return np.ascontiguousarray(a)


def fm_view(image, support, out=(1520, 912)):
    # Mammo-FM protocol on the shared soft base: bbox of the tissue support, stretch to 1520x912.
    ys, xs = np.where(support)
    if not len(ys):
        raise ValueError('Empty support mask.')
    crop = image[ys.min():ys.max() + 1, :xs.max() + 1]
    return cv2.resize(np.ascontiguousarray(crop), (out[1], out[0]), interpolation=cv2.INTER_AREA)


def robust_scale(array, pct=99.8, border=8):
    # Black point stays 0 (relative density kept); white point = tissue percentile away from the view border,
    # so a bright edge strip no longer squeezes dark images (EDA 31: max pixel sits on the border in 91-100% of views).
    inner = np.zeros(array.shape, bool)
    inner[border:-border, border:-border] = True
    tissue = array[(array > 0) & inner]
    hi = np.percentile(tissue, pct) if tissue.size else 255.0
    return np.where(array > 0, np.clip(array.astype(np.float32) / max(hi, 1.0) * 255, 1, 255), 0).astype(np.uint8)


def tissue_crop(image, support):
    # Bounding box of the tissue support, chest wall kept at x = 0 (DINOv3 view, aspect preserved by the later letterbox).
    ys, xs = np.where(support)
    if not len(ys):
        raise ValueError('Empty support mask.')
    return np.ascontiguousarray(image[ys.min():ys.max() + 1, :xs.max() + 1])
