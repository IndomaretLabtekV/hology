"""Final v7 pipeline check, exactly the functions the notebook embeds:
shared soft base (harmonize q75, decode 1/2, flip, soft mask) -> DINOv3 view (letterbox 1024x768) and Mammo-FM view (bbox + stretch 1520x912),
then the selected augmentation (cover rotation +-10, masked jitter +-0.1, vflip). Saves per-class sheets + all-image FM sheet."""
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from deep_common import CLASSES, grid, manifest, output
from prep_v6 import preprocess
from prep_v7 import augment_v7, fm_view


def views(path):
    img, mask, support, flipped = preprocess(path, (2048, 1536), 'soft', True)
    dino = cv2.resize(img, (768, 1024), interpolation=cv2.INTER_AREA)
    return dino, fm_view(img, support)


def main():
    out = output(29)
    data = manifest()
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(views, data.path))
    rows = []
    for r, (dino, fm) in zip(data.itertuples(), res):
        t_d, t_f = dino[dino > 0], fm[fm > 0]
        rows.append(dict(image_id=r.image_id, label=r.label, split=r.split,
                         dino_fill=float((dino > 0).mean()), fm_fill=float((fm > 0).mean()),
                         fm_left_touch=float((fm[:, :4] > 0).any(1).mean()), fm_black_right=float((fm[:, -8:] == 0).all(1).mean()),
                         tissue_median_dino=float(np.median(t_d)), tissue_median_fm=float(np.median(t_f))))
    d = pd.DataFrame(rows)
    d.to_csv(out / 'view_stats.csv', index=False)
    print(d.groupby(['split', 'label'])[['dino_fill', 'fm_fill', 'fm_left_touch', 'fm_black_right']].mean().round(3).to_string())
    bad = d[(d.fm_left_touch < .3) | (d.fm_fill < .35)]
    print('\nFM views to eyeball (weak chest-wall contact or low fill):', len(bad), bad.image_id.str[6:10].tolist())
    rng = np.random.default_rng(0)
    by_id = dict(zip(data.image_id, res))
    ims, titles = [], []
    for c in CLASSES:
        for iid in data[data.label == c].sample(3, random_state=3).image_id:
            dino, fm = by_id[iid]
            fm_small = cv2.resize(fm, (456, 760), interpolation=cv2.INTER_AREA)
            ims += [dino, augment_v7(dino, rng), fm_small, augment_v7(fm_small, rng)]
            titles += [f'{iid[6:10]} {c} DINO', 'DINO aug', 'FM view', 'FM aug']
    grid(ims, titles, out / 'views_per_class.png', cols=6)
    for start in range(0, len(data), 72):
        chunk = data.iloc[start:start + 72]
        grid([cv2.resize(by_id[i][1], (228, 380), interpolation=cv2.INTER_AREA) for i in chunk.image_id],
             [f'{i[6:10]} {l}' for i, l in zip(chunk.image_id, chunk.label)], out / f'fm_all_{start:03d}.png', cols=12)
    if len(bad):
        grid([cv2.resize(by_id[i][1], (228, 380)) for i in bad.image_id[:24]], bad.image_id.str[6:10].tolist()[:24], out / 'fm_suspects.png')


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
