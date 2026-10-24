"""v6 pipeline check: soft base -> Mammo-FM view -> robust_scale -> augment_v7 (cover rotation +-7, masked jitter, vflip).
Per-class sheets, all-image contact sheets, and intensity parity between classes/splits."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deep_common import CLASSES, grid, manifest, output
from prep_v6 import preprocess
from prep_v7 import augment_v7, fm_view, robust_scale


def view(path):
    img, mask, support, _ = preprocess(path, (2048, 1536), 'soft', True)
    return robust_scale(fm_view(img, support))


def main():
    out = output(33)
    data = manifest()
    with ThreadPoolExecutor(6) as pool:
        views = dict(zip(data.image_id, pool.map(view, data.path)))
    rows = []
    for r in data.itertuples():
        v = views[r.image_id]; t = v[v > 0]
        rows.append(dict(image_id=r.image_id, label=r.label, split=r.split, p50=np.median(t), p05=np.percentile(t, 5),
                         sat=(t >= 254).mean(), fill=(v > 0).mean(), bg_nonzero=(v[:, -16:] > 0).mean()))
    d = pd.DataFrame(rows); d.to_csv(out / 'v6_view_stats.csv', index=False)
    print(d.groupby(['split', 'label'])[['p50', 'p05', 'sat', 'fill', 'bg_nonzero']].median().round(3).to_string())
    tr = d[d.split == 'train']; y = tr.label.map(CLASSES.index)
    print('AUC p50 Normal vs rest %.3f | M vs B %.3f' % (roc_auc_score(y == 0, tr.p50), roc_auc_score(y[y > 0] == 2, tr.p50[y > 0])))
    print('darkest after robust_scale:', d.nsmallest(5, 'p50')[['image_id', 'label', 'p50']].values.tolist())
    rng = np.random.default_rng(0)
    ims, titles = [], []
    for c in CLASSES:
        for iid in data[data.label == c].sample(3, random_state=11).image_id:
            v = cv2.resize(views[iid], (456, 760), interpolation=cv2.INTER_AREA)
            ims += [v] + [augment_v7(v, rng) for _ in range(3)]
            titles += [f'{iid[6:10]} {c} v6 view'] + ['aug'] * 3
    grid(ims, titles, out / 'v6_views_per_class.png', cols=8)
    for start in range(0, len(data), 72):
        chunk = data.iloc[start:start + 72]
        grid([cv2.resize(views[i], (228, 380), interpolation=cv2.INTER_AREA) for i in chunk.image_id],
             [f'{i[6:10]} {l}' for i, l in zip(chunk.image_id, chunk.label)], out / f'v6_all_{start:03d}.png', cols=12)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
