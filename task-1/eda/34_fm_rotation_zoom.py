"""Cover rotation on the Mammo-FM view (tall 1.67 aspect, tissue fills the bbox): how much tissue and how much of the
nipple/skin edge (rightmost tissue) is cut per max angle? The DINO letterbox view (EDA 26) had spare black canvas; this one does not."""
import cv2
import numpy as np
import pandas as pd

from deep_common import CLASSES, grid, manifest, output
from prep_v6 import preprocess
from prep_v7 import cover_scale, fm_view, robust_scale, rotate_cover


def main():
    out = output(34)
    data = manifest(); tr = data[data.split == 'train']
    rng = np.random.default_rng(0)
    rows, cache = [], {}
    for r in tr.sample(60, random_state=0).itertuples():
        img, mask, support, _ = preprocess(r.path, (1024, 768), 'soft', True)
        v = cv2.resize(robust_scale(fm_view(img, support)), (228, 380), interpolation=cv2.INTER_AREA)
        cache[r.image_id] = (v, r.label)
        tissue = (v > 0).astype(np.uint8)
        right = tissue.shape[1] - 1 - np.argmax(tissue[:, ::-1], 1)
        tip = np.zeros_like(tissue); rows_t = tissue.any(1)
        tip[np.flatnonzero(rows_t), right[rows_t]] = 1
        tip = cv2.dilate(tip, np.ones((3, 3), np.uint8))
        for deg in [3, 5, 7]:
            for angle in rng.uniform(-deg, deg, 4):
                t = rotate_cover(tissue, angle, True) > 0; s = cover_scale(v.shape, angle)
                edge = rotate_cover(tip, angle, True) > 0
                rows.append(dict(image_id=r.image_id, label=r.label, deg=deg, zoom=s, tissue_kept=t.sum() / s ** 2 / tissue.sum(),
                                 skin_edge_kept=edge.sum() / s / max(tip.sum(), 1)))
    d = pd.DataFrame(rows); d.to_csv(out / 'fm_rotation_zoom.csv', index=False)
    print(d.groupby('deg')[['zoom', 'tissue_kept', 'skin_edge_kept']].agg(['mean', 'min']).round(3).to_string())
    print(d.groupby(['deg', 'label'])[['tissue_kept', 'skin_edge_kept']].mean().round(3).unstack().to_string())
    ims, titles = [], []
    for c in CLASSES:
        iid = next(i for i, (v, l) in cache.items() if l == c)
        v = cache[iid][0]
        ims += [v, rotate_cover(v, 3), rotate_cover(v, -3), rotate_cover(v, 7), rotate_cover(v, -7)]
        titles += [f'{iid[6:10]} {c}', 'cover +3', 'cover -3', 'cover +7', 'cover -7']
    grid(ims, titles, out / 'fm_rotation_zoom.png', cols=5)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
