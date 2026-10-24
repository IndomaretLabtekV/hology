"""ColorJitter as used in torchvision vs masked brightness/contrast (prep_v7.masked_jitter), per class, on the soft cache.
Checks: background lift, clipping of bright tissue (calcifications/masses), collapse of dark images."""
import cv2
import numpy as np
import pandas as pd

from deep_common import CLASSES, ROOT, grid, manifest, output, plt
from prep_v7 import masked_jitter


def torch_jitter(a, b, c):
    # torchvision ColorJitter on grayscale: brightness scales, contrast blends with the global image mean
    f = np.clip(a.astype(np.float32) * b, 0, 255)
    return np.clip(c * f + (1 - c) * f.mean(), 0, 255).astype(np.uint8)


def measure(a, z):
    tissue = a > 0
    bright = a >= np.percentile(a[tissue], 99)
    return dict(bg_lift=float((z[~tissue] > 2).mean()),
                bright_clipped=float((z[bright] >= 254).mean() - (a[bright] >= 254).mean()),
                tissue_std_ratio=float(z[tissue].std() / max(a[tissue].std(), 1e-6)),
                tissue_mae=float(np.abs(z[tissue].astype(float) - a[tissue]).mean()))


def main():
    out = output(27)
    data = manifest()
    tr = data[data.split == 'train']
    load = lambda iid: cv2.imread(str(ROOT / 'outputs/20/cache_soft' / iid.replace('.jpg', '.png')), 0)
    rng = np.random.default_rng(42)
    rows = []
    for r in tr.itertuples():
        a = cv2.resize(load(r.image_id), (384, 512), interpolation=cv2.INTER_AREA)
        for name, lim in [('torch_0.2', .2), ('torch_0.4', .4), ('masked_0.1', .1), ('masked_0.2', .2)]:
            for _ in range(3):
                b, c = rng.uniform(1 - lim, 1 + lim, 2)
                z = torch_jitter(a, b, c) if name.startswith('torch') else masked_jitter(a, b, c)
                rows.append(dict(image_id=r.image_id, label=r.label, variant=name, **measure(a, z)))
    d = pd.DataFrame(rows)
    d.to_csv(out / 'jitter_metrics.csv', index=False)
    print(d.groupby('variant')[['bg_lift', 'bright_clipped', 'tissue_std_ratio', 'tissue_mae']].agg(['mean', 'max']).round(4).to_string())
    print('\nmasked_0.1 by class:\n', d[d.variant == 'masked_0.1'].groupby('label')[['bright_clipped', 'tissue_mae']].mean().round(4).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ['bg_lift', 'bright_clipped', 'tissue_std_ratio']):
        d.groupby('variant')[col].mean().plot.bar(ax=ax); ax.set_title(col)
    fig.tight_layout(); fig.savefig(out / 'jitter_metrics.png', dpi=110); plt.close(fig)
    picks = []
    for c in CLASSES:
        picks += tr[tr.label == c].sample(2, random_state=1).image_id.tolist()
    picks += ['image_0106.jpg', 'image_0101.jpg']
    ims, titles = [], []
    for iid in picks:
        a = load(iid); lab = tr.set_index('image_id').label[iid]
        for t, z in [('original', a), ('torch b1.2 c0.7', torch_jitter(a, 1.2, .7)), ('torch b0.8 c1.3', torch_jitter(a, .8, 1.3)),
                     ('masked b1.1 c0.9', masked_jitter(a, 1.1, .9)), ('masked b0.9 c1.1', masked_jitter(a, .9, 1.1))]:
            ims.append(z); titles.append(f'{iid[6:10]} {lab} {t}')
    grid(ims, titles, out / 'jitter_per_class.png', cols=5)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
