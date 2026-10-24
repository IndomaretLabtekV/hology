"""Quantify geometry, retained tissue, background and intensity changes. Not a classifier benchmark."""
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from deep_common import grid, manifest, output, plt, samples
from prep_v6 import prepare_array, read_gray


def measure(row):
    raw = read_gray(row.path, 2, harmonize=True)
    base, mask, support, _ = prepare_array(raw, variant='raw')
    records = []
    for variant in ['raw', 'hard', 'soft', 'gamma', 'clahe', 'crop']:
        image, m, s, _ = prepare_array(raw, variant=variant)
        tissue = image[m]
        far = image[~s]
        records.append(dict(image_id=row.image_id, label=row.label, split=row.split, variant=variant,
                            mask_fraction=m.mean(), far_bg_std=far.std() if len(far) else 0,
                            far_bg_nonzero=(far > 0).mean() if len(far) else 0,
                            tissue_low=(tissue < 5).mean(), tissue_high=(tissue > 250).mean(),
                            tissue_p50=np.median(tissue), tissue_range=np.percentile(tissue, 99) - np.percentile(tissue, 1),
                            mae_tissue=float(np.abs(image.astype(float) - base)[mask].mean()) if variant != 'crop' else np.nan))
    # Trimming removes original tissue, assessed against the untrimmed reference mask.
    for trim in [0.005, 0.01]:
        h, w = mask.shape; dy, dx = round(h * trim), round(w * trim)
        keep = np.zeros_like(mask); keep[dy:h-dy, dx:w-dx] = True
        records.append(dict(image_id=row.image_id, label=row.label, split=row.split, variant=f'trim_{trim}',
                            tissue_removed=float((mask & ~keep).sum() / max(mask.sum(), 1))))
    cache = output(20) / 'cache_soft'
    image, _, _, _ = prepare_array(raw)
    assert image.shape == (1024, 768) and image.dtype == np.uint8
    if not cv2.imwrite(str(cache / row.image_id.replace('.jpg', '.png')), image):
        raise OSError('cache write failed')
    return records


def main():
    out = output(20); (out / 'cache_soft').mkdir(exist_ok=True)
    data = manifest()
    with ThreadPoolExecutor(4) as pool:
        rows = [r for group in pool.map(measure, data.itertuples(index=False)) for r in group]
    res = pd.DataFrame(rows); res.to_csv(out / 'preprocessing_metrics.csv', index=False)
    cols = ['mask_fraction', 'far_bg_std', 'far_bg_nonzero', 'tissue_low', 'tissue_high', 'tissue_range', 'mae_tissue', 'tissue_removed']
    summary = res.groupby('variant')[cols].mean()
    summary.to_csv(out / 'summary.csv'); print(summary.round(5).to_string())
    picks = samples(data)
    for row in picks.itertuples():
        raw = read_gray(row.path, 2, harmonize=True)
        variants = ['raw', 'hard', 'soft', 'gamma', 'clahe', 'crop']
        ims = [prepare_array(raw, variant=v)[0] for v in variants]
        grid(ims, [f'{row.label} {row.image_id}\n{v}' for v in variants], out / f'variants_{row.image_id[:-4]}.png')
    # All-image contact sheets make low-mask failures inspectable instead of reporting only means.
    for split in ['train', 'test']:
        part = data[data.split == split]
        ims = [cv2.imread(str(out / 'cache_soft' / i.replace('.jpg', '.png')), 0)[::4, ::4] for i in part.image_id]
        for start in range(0, len(part), 54):
            sub = part.iloc[start:start+54]
            grid(ims[start:start+54], [f'{r.image_id}\n{r.label}' for r in sub.itertuples()], out / f'all_{split}_{start:03d}.png', cols=9)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ['mae_tissue', 'far_bg_std', 'tissue_high']):
        summary[col].dropna().plot.bar(ax=ax); ax.set_title(col)
    fig.tight_layout(); fig.savefig(out / 'preprocessing_comparison.png', dpi=120); plt.close(fig)
    decision = dict(candidate='soft', image_height=1024, image_width=768, trim=0, harmonize=True,
                    reason='Preserve full field and original tissue intensities; remove exterior texture with a smooth transition.',
                    limitation='Mask is heuristic; semantic preservation, residual in-tissue shortcuts and macro F1 need GPU validation.')
    (out / 'decision.json').write_text(json.dumps(decision, indent=2))


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
