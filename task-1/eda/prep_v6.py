"""CPU image operations shared verbatim with the v6 notebook; no model inference."""
import io

import cv2
import numpy as np
from PIL import Image

from prep import breast_mask


def read_gray(path, scale=2, harmonize=False):
    with Image.open(path) as image:
        if harmonize:
            buf = io.BytesIO()
            image.convert('L').save(buf, 'JPEG', quality=75)
            buf.seek(0)
            with Image.open(buf) as encoded:
                encoded.draft('L', (encoded.width // scale, encoded.height // scale))
                return np.array(encoded.convert('L'))
        image.draft('L', (image.width // scale, image.height // scale))
        return np.array(image.convert('L'))


def letterbox(array, out=(1024, 768), nearest=False):
    h, w = array.shape
    ratio = min(out[0] / h, out[1] / w)
    hh, ww = max(1, round(h * ratio)), max(1, round(w * ratio))
    resized = cv2.resize(array, (ww, hh), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
    result = np.zeros(out, dtype=array.dtype)
    top = (out[0] - hh) // 2
    result[top:top + hh, :ww] = resized
    return result


def prepare_array(raw, out=(1024, 768), variant='soft', trim=0.0):
    if raw.ndim != 2 or raw.dtype != np.uint8 or min(raw.shape) < 32:
        raise ValueError('Expected a grayscale uint8 image at least 32x32.')
    h, w = raw.shape
    dy, dx = round(h * trim), round(w * trim)
    a = raw[dy:h - dy, dx:w - dx].copy()
    # The existing mask was tuned at quarter-resolution, approximately 1185px tall.
    sh = min(1185, a.shape[0])
    sw = max(32, round(a.shape[1] * sh / a.shape[0]))
    small = cv2.resize(a, (sw, sh), interpolation=cv2.INTER_AREA)
    mask_small = breast_mask(small)
    mask = cv2.resize(mask_small.astype(np.uint8), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    flipped = mask[:, :a.shape[1] // 2].sum() < mask[:, a.shape[1] // 2:].sum()
    if flipped:
        a, mask = a[:, ::-1].copy(), mask[:, ::-1].copy()
    k = max(3, round(15 * a.shape[0] / 1185))
    support = cv2.dilate(mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    distance = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(1 - distance / k, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    if variant == 'raw':
        processed = a
    elif variant == 'hard':
        processed = a * mask
    elif variant in ('soft', 'gamma', 'clahe', 'crop'):
        processed = a.copy()
        if variant == 'gamma':
            processed = np.round((a.astype(np.float32) / 255) ** 0.8 * 255).astype(np.uint8)
        elif variant == 'clahe':
            processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(a)
        processed = np.round(processed * alpha).astype(np.uint8)
    else:
        raise ValueError(variant)
    if variant == 'crop':
        yy, xx = np.where(support)
        if not len(yy):
            raise ValueError('Empty support mask.')
        sl = np.s_[yy.min():yy.max() + 1, :xx.max() + 1]
        processed, mask, support = processed[sl], mask[sl], support[sl]
    image = letterbox(processed, out)
    m = letterbox(mask, out, nearest=True).astype(bool)
    s = letterbox(support, out, nearest=True).astype(bool)
    return image, m, s, bool(flipped)


def preprocess(path, out=(1024, 768), variant='soft', harmonize=True):
    return prepare_array(read_gray(path, scale=2, harmonize=harmonize), out, variant)


def rotation_matrix(shape, angle, fit=False):
    h, w = shape
    rad = np.deg2rad(angle)
    factor = min((w - 1) / (abs((w - 1) * np.cos(rad)) + abs((h - 1) * np.sin(rad))),
                 (h - 1) / (abs((h - 1) * np.cos(rad)) + abs((w - 1) * np.sin(rad)))) if fit else 1.0
    return cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, factor)


def rotate(array, angle, border='reflect', fit=False, nearest=False):
    if border not in ('reflect', 'constant'):
        raise ValueError(border)
    h, w = array.shape
    return cv2.warpAffine(array, rotation_matrix(array.shape, angle, fit), (w, h),
                         flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101 if border == 'reflect' else cv2.BORDER_CONSTANT)


def gamma_intensity(array, gamma):
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    return np.round((array.astype(np.float32) / 255) ** gamma * 255).astype(np.uint8)


def augment(array, rng, degrees=5.0, gamma_range=(0.9, 1.1)):
    # ponytail: fit rotation shrinks the entire field; avoid larger angles or use GPU ablation before widening.
    angle = float(rng.uniform(-degrees, degrees))
    return gamma_intensity(rotate(array, angle, border='constant', fit=True), float(rng.uniform(*gamma_range)))
