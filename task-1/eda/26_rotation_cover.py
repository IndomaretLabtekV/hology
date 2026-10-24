"""Random rotation, measured and viewed per class on the soft cache (outputs/20/cache_soft, 1024x768).
Question: which rotation keeps the chest wall attached to the canvas edge, adds no fake tissue and no black wedge?"""
import cv2
import numpy as np
import pandas as pd

from deep_common import CLASSES, ROOT, grid, manifest, output, plt
from prep_v6 import rotate
from prep_v7 import cover_scale, rotate_cover

VARIANTS = ['naive', 'reflect', 'fit_constant', 'cover']


def apply(name, a, angle, nearest=False):
    if name == 'naive':
        return rotate(a, angle, 'constant', False, nearest)
    if name == 'reflect':
        return rotate(a, angle, 'reflect', False, nearest)
    if name == 'fit_constant':
        return rotate(a, angle, 'constant', True, nearest)
    return rotate_cover(a, angle, nearest)


def chest_gap(tissue):
    # rows whose tissue exists but does not start at the left canvas edge = chest wall detached
    rows = tissue.any(1)
    first = np.argmax(tissue, 1)
    return float(((first > 3) & rows).sum() / max(rows.sum(), 1))


def main():
    out = output(26)
    data = manifest()
    tr = data[data.split == 'train']
    load = lambda iid: cv2.imread(str(ROOT / 'outputs/20/cache_soft' / iid.replace('.jpg', '.png')), 0)
    rng = np.random.default_rng(42)
    rows = []
    for r in tr.itertuples():
        a = cv2.resize(load(r.image_id), (384, 512), interpolation=cv2.INTER_AREA)
        tissue = (a > 10).astype(np.uint8)
        base_gap = chest_gap(tissue > 0)
        for deg in [5, 10, 15]:
            for angle in rng.uniform(-deg, deg, 2):
                ones = np.ones_like(tissue)
                for name in VARIANTS:
                    t = apply(name, tissue, angle, True) > 0
                    valid = apply(name, ones, angle, True) > 0
                    inside = rotate(ones, angle, 'constant', False, True) > 0
                    mirrored = t & ~inside if name == 'reflect' else np.zeros_like(t)
                    zoom = cover_scale(a.shape, angle) if name == 'cover' else 1.0
                    rows.append(dict(image_id=r.image_id, label=r.label, deg=deg, variant=name,
                                     invalid_canvas=float((~valid).mean()) if name != 'reflect' else 0.0,
                                     fake_tissue=float(mirrored.sum() / max(tissue.sum(), 1)),
                                     tissue_kept=float((t & ~mirrored).sum() / zoom ** 2 / max(tissue.sum(), 1)),
                                     chest_gap=chest_gap(t & valid) - base_gap,
                                     zoom=zoom))
    d = pd.DataFrame(rows)
    d.to_csv(out / 'rotation_metrics.csv', index=False)
    s = d.groupby(['deg', 'variant'])[['invalid_canvas', 'fake_tissue', 'tissue_kept', 'chest_gap', 'zoom']].mean()
    print(s.round(4).to_string())
    print('\nPer class, cover 10 deg:\n', d[(d.variant == 'cover') & (d.deg == 10)].groupby('label')[['tissue_kept', 'chest_gap', 'zoom']].mean().round(4).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, ['invalid_canvas', 'chest_gap', 'tissue_kept']):
        s[col].unstack().plot.bar(ax=ax); ax.set_title(col)
    fig.tight_layout(); fig.savefig(out / 'rotation_metrics.png', dpi=110); plt.close(fig)
    picks = []
    for c in CLASSES:
        picks += tr[tr.label == c].sample(2, random_state=1).image_id.tolist()
    picks += ['image_0106.jpg', 'image_0153.jpg']
    ims, titles = [], []
    for iid in picks:
        a = load(iid); lab = tr.set_index('image_id').label[iid]
        ims.append(a); titles.append(f'{iid[6:10]} {lab} original')
        for name in VARIANTS:
            ims.append(apply(name, a, 12.0)); titles.append(f'{name} +12')
    grid(ims, titles, out / 'rotation_per_class.png', cols=5)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
