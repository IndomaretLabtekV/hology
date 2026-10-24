"""Augmentation on the chosen DINOv3 view (tissue crop, robust scale, letterbox 1344x720):
cover rotation +-3/+-5/+-7 (tissue and skin-edge kept), masked jitter (background stays 0), vflip. Per-class sheets."""
import cv2
import numpy as np
import pandas as pd

from deep_common import CLASSES, grid, manifest, output
from prep_v6 import letterbox, preprocess
from prep_v7 import augment_v7, cover_scale, masked_jitter, robust_scale, rotate_cover

SIZE = (1344, 720)


def dino_view(path):
    img, mask, support, _ = preprocess(path, (2368, 1776), 'soft', True)
    ys, xs = np.where(support)
    return letterbox(robust_scale(np.ascontiguousarray(img[ys.min():ys.max() + 1, :xs.max() + 1])), SIZE)


def main():
    out = output(40)
    data = manifest(); tr = data[data.split == 'train']
    rng = np.random.default_rng(0)
    rows, cache = [], {}
    for r in tr.sample(60, random_state=1).itertuples():
        v = cv2.resize(dino_view(r.path), (360, 672), interpolation=cv2.INTER_AREA)
        cache[r.image_id] = (v, r.label)
        tissue = (v > 0).astype(np.uint8)
        right = tissue.shape[1] - 1 - np.argmax(tissue[:, ::-1], 1)
        rows_t = tissue.any(1)
        tip = np.zeros_like(tissue); tip[np.flatnonzero(rows_t), right[rows_t]] = 1
        tip = cv2.dilate(tip, np.ones((3, 3), np.uint8))
        for deg in [3, 5, 7]:
            for angle in rng.uniform(-deg, deg, 4):
                s = cover_scale(v.shape, angle)
                rows.append(dict(label=r.label, deg=deg, zoom=s,
                                 tissue_kept=(rotate_cover(tissue, angle, True) > 0).sum() / s ** 2 / tissue.sum(),
                                 skin_edge_kept=(rotate_cover(tip, angle, True) > 0).sum() / s / max(tip.sum(), 1)))
        j = masked_jitter(v, 1.1, 0.9)
        rows.append(dict(label=r.label, deg=-1, bg_lift=float((j[v == 0] > 0).mean()), clipped=float((j[v > 0] >= 255).mean())))
    d = pd.DataFrame(rows); d.to_csv(out / 'dino_augment.csv', index=False)
    print(d[d.deg > 0].groupby('deg')[['zoom', 'tissue_kept', 'skin_edge_kept']].agg(['mean', 'min']).round(3).to_string())
    print('masked jitter: background lift', d[d.deg == -1].bg_lift.mean().round(4), '| tissue clipped', d[d.deg == -1].clipped.mean().round(4))
    ims, titles = [], []
    for c in CLASSES:
        iid = next(i for i, (v, l) in cache.items() if l == c)
        v = cache[iid][0]
        ims += [v, rotate_cover(v, 3), rotate_cover(v, -7)] + [augment_v7(v, rng, 3.0) for _ in range(3)]
        titles += [f'{iid[6:10]} {c}', 'cover +3', 'cover -7', 'aug', 'aug', 'aug']
    grid(ims, titles, out / 'dino_augment_per_class.png', cols=6)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
