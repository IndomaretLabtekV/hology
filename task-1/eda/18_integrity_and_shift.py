"""Full decode integrity, exact duplicates, file confounding, and train/test image quality."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from deep_common import CLASSES, grid, manifest, output, plt
from prep_v6 import prepare_array


def inspect(row):
    with Image.open(row.path) as im:
        mode, w, h = im.mode, im.width, im.height
        quant = float(np.mean(im.quantization[0]))
        # Full decode verifies the payload; hashing raw pixels separates recompression from byte equality.
        gray = np.array(im.convert('L'))
    pixel_hash = hashlib.sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()
    small = cv2.resize(gray, (384, 512), interpolation=cv2.INTER_AREA)
    image, mask, support, flipped = prepare_array(small, (512, 384))
    tissue = image[mask]
    aligned = small[:, ::-1] if flipped else small
    far = aligned[~support]
    return dict(image_id=row.image_id, width=w, height=h, mode=mode, q_mean=quant,
                file_sha256=hashlib.sha256(row.path.read_bytes()).hexdigest(), pixel_sha256=pixel_hash,
                bytes=row.path.stat().st_size, mask_fraction=float(mask.mean()), flipped=flipped,
                tissue_p01=float(np.percentile(tissue, 1)), tissue_median=float(np.median(tissue)),
                tissue_p99=float(np.percentile(tissue, 99)), tissue_std=float(tissue.std()),
                tissue_black=float((tissue == 0).mean()), tissue_white=float((tissue == 255).mean()),
                bg_median=float(np.median(far)) if len(far) else np.nan,
                lap_std=float(cv2.Laplacian(image, cv2.CV_32F)[mask].std()))


def main():
    out = output(18)
    data = manifest()
    with ThreadPoolExecutor(4) as pool:
        stats = data.drop(columns='path').merge(pd.DataFrame(pool.map(inspect, data.itertuples(index=False))), on='image_id', validate='one_to_one')
    stats.to_csv(out / 'integrity.csv', index=False)
    duplicates = stats[stats.duplicated('pixel_sha256', keep=False)].sort_values('pixel_sha256')
    duplicates.to_csv(out / 'exact_duplicates.csv', index=False)
    numeric = ['tissue_median', 'tissue_std', 'mask_fraction', 'bg_median', 'lap_std']
    rows = []
    for f in numeric:
        a, b = stats.loc[stats.split == 'train', f], stats.loc[stats.split == 'test', f]
        ks = ks_2samp(a, b)
        rows.append(dict(feature=f, train_median=a.median(), test_median=b.median(), ks=ks.statistic, p_unadjusted=ks.pvalue))
    shifts = pd.DataFrame(rows)
    shifts.to_csv(out / 'shift.csv', index=False)
    train = stats[stats.split == 'train']
    auc = roc_auc_score(train.label == 'Normal', train['mode'] == 'L')
    summary = dict(decoded=len(stats), train=sum(stats.split == 'train'), test=sum(stats.split == 'test'),
                   exact_duplicate_images=len(duplicates), normal_mode_auc=auc,
                   interpretation='KS is exploratory, image-level and class-mix confounded; pixels have no verified physical spacing.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(pd.crosstab(stats.label, stats['mode']).to_string())
    print(shifts.round(4).to_string(index=False))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.flat, ['q_mean'] + numeric):
        for label in CLASSES + ['?']:
            vals = stats.loc[stats.label == label, f]
            ax.hist(vals, bins=15, alpha=0.45, density=True, label=label)
        ax.set_title(f)
    axes.flat[0].legend()
    fig.tight_layout(); fig.savefig(out / 'quality_and_shift.png', dpi=120); plt.close(fig)
    hard = train.sort_values('tissue_median').head(12).merge(data[['image_id', 'path']], on='image_id')
    images = []
    for p in hard.path:
        with Image.open(p) as im:
            im.draft('L', (384, 512)); images.append(np.array(im.convert('L')))
    grid(images, [f'{r.image_id}\n{r.label}, median={r.tissue_median:.0f}' for r in hard.itertuples()], out / 'dark_train_cases.png')


if __name__ == '__main__':
    main()
