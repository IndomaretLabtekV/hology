"""DINOv3 (ViT, patch 16) input design. A ViT spends the same compute on every 16x16 patch, background included.
For each candidate view and resolution: fraction of patch tokens that hold tissue, tissue pixels per token (detail),
aspect distortion, and how many 16x16 tokens a small bright dot / mass covers. Per-class sample sheets."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from deep_common import CLASSES, grid, manifest, output, plt
from prep_v6 import letterbox, preprocess
from prep_v7 import robust_scale

SIZES = [(768, 576), (1024, 768), (1280, 960), (1344, 720), (1536, 832)]


def crop_box(image, support):
    ys, xs = np.where(support)
    return np.ascontiguousarray(image[ys.min():ys.max() + 1, :xs.max() + 1])


def views(path):
    img, mask, support, _ = preprocess(path, (2368, 1776), 'soft', True)
    crop = crop_box(img, support)
    return {'full': img, 'crop': crop}, crop.shape[0] / crop.shape[1]


def token_stats(a, size):
    v = letterbox(a, size)
    t = (v > 0).reshape(size[0] // 16, 16, size[1] // 16, 16).mean((1, 3))
    return float((t > 0.5).mean()), float((v > 0).sum() / max((t > 0.5).sum(), 1))


def main():
    out = output(39)
    data = manifest()
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(views, data.path))
    rows = []
    for r, (v, aspect) in zip(data.itertuples(), res):
        row = dict(image_id=r.image_id, label=r.label, split=r.split, crop_aspect=aspect)
        for name, a in v.items():
            for size in SIZES:
                frac, _ = token_stats(a, size)
                scale = min(size[0] / a.shape[0], size[1] / a.shape[1])
                row[f'{name}_{size[0]}_tissue_tokens'] = frac
                # a 1 mm-ish dot at decode 1/2 is ~5 px; after scaling how many px / tokens does it span
                row[f'{name}_{size[0]}_scale'] = scale
        rows.append(row)
    d = pd.DataFrame(rows); d.to_csv(out / 'dino_view_tokens.csv', index=False)
    cols = [c for c in d.columns if c.endswith('tissue_tokens') or c.endswith('scale')]
    print(d[cols].median().round(3).to_string())
    print('\ncrop aspect (H/W) median by class:', d.groupby('label').crop_aspect.median().round(3).to_dict(),
          '| 4:3 canvas aspect = 1.333')
    print('tissue-token fraction by class (crop_1024):', d.groupby('label').crop_1024_tissue_tokens.median().round(3).to_dict())
    for name in ['full', 'crop']:
        for size in SIZES:
            s = d[f'{name}_{size[0]}_scale'].median()
            tok = size[0] // 16 * size[1] // 16
            print(f'{name:4s} {size[0]}x{size[1]}: tokens {tok}, tissue tokens {d[f"{name}_{size[0]}_tissue_tokens"].median():.2f}, '
                  f'scale vs decode-1/2 {s:.3f}, 5px dot -> {5 * s:.1f}px = {5 * s / 16:.2f} token')
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ['full', 'crop']:
        ax.plot([s[0] for s in SIZES], [d[f'{name}_{s[0]}_tissue_tokens'].median() for s in SIZES], marker='o', label=name)
    ax.set_xlabel('input height'); ax.set_ylabel('median fraction of tissue tokens'); ax.legend(); ax.set_title('DINOv3 token budget on tissue')
    fig.tight_layout(); fig.savefig(out / 'dino_token_budget.png', dpi=110); plt.close(fig)
    ims, titles = [], []
    by_id = dict(zip(data.image_id, [r[0] for r in res]))
    for c in CLASSES:
        for iid in data[data.label == c].sample(2, random_state=21).image_id:
            v = by_id[iid]
            ims += [letterbox(v['full'], (1024, 768)), letterbox(robust_scale(v['crop']), (1024, 768)), letterbox(robust_scale(v['crop']), (1344, 720))]
            titles += [f'{iid[6:10]} {c} full 1024x768', 'crop 1024x768', 'crop 1344x720']
    grid(ims, titles, out / 'dino_views_per_class.png', cols=6)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
