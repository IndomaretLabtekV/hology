"""Do microcalcification-like bright dots survive the Mammo-FM input size?
Detect small bright blobs (white top-hat) on the tissue bbox at decode 1/2 (~native/2), then measure how much
top-hat contrast survives a down-up resample to 1520x912 (v5), 2048x1228 and 2560x1536. Also: blob counts per class."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deep_common import CLASSES, grid, manifest, output, plt
from prep_v6 import read_gray, prepare_array

SIZES = {'1520x912': (1520, 912), '2048x1228': (2048, 1228), '2560x1536': (2560, 1536)}
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def tophat(a):
    return cv2.morphologyEx(a, cv2.MORPH_TOPHAT, KERNEL).astype(np.float32)


def analyse(r):
    raw = read_gray(r.path, 2, True)
    img, mask, support, _ = prepare_array(raw, raw.shape, 'soft')
    ys, xs = np.where(support)
    crop = np.ascontiguousarray(img[ys.min():ys.max() + 1, :xs.max() + 1])
    inner = cv2.erode(mask[ys.min():ys.max() + 1, :xs.max() + 1].astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    th = tophat(crop)
    thr = max(20.0, float(np.percentile(th[inner], 99.95))) if inner.any() else 255
    n, lab, st, cen = cv2.connectedComponentsWithStats(((th > thr) & inner).astype(np.uint8), 8)
    keep = [k for k in range(1, n) if 2 <= st[k, cv2.CC_STAT_AREA] <= 60]
    peaks = [(int(cen[k][1]), int(cen[k][0])) for k in keep]
    row = dict(image_id=r.image_id, label=r.label, split=r.split, crop_h=crop.shape[0], crop_w=crop.shape[1], n_blobs=len(keep),
               blob_area_median=float(np.median([st[k, cv2.CC_STAT_AREA] for k in keep])) if keep else np.nan)
    for name, (h, w) in SIZES.items():
        small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
        back = cv2.resize(small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        tb = tophat(back)
        row[f'keep_{name}'] = float(np.mean([tb[p] / max(th[p], 1) for p in peaks])) if peaks else np.nan
        row[f'scale_{name}'] = h / crop.shape[0]
    return row, (crop, peaks)


def main():
    out = output(32)
    data = manifest()
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(analyse, data.itertuples(index=False)))
    d = pd.DataFrame([r[0] for r in res]); crops = {r[0]['image_id']: r[1] for r in res}
    d.to_csv(out / 'calc_resolution.csv', index=False)
    keep_cols = [f'keep_{k}' for k in SIZES]
    print(d.groupby(['split', 'label'])[['crop_h', 'n_blobs', 'blob_area_median'] + keep_cols + ['scale_1520x912']].median().round(3).to_string())
    tr = d[d.split == 'train']; y = tr.label.map(CLASSES.index)
    print('\nAUC n_blobs: Malignant vs rest %.3f | M vs B %.3f | Normal vs rest %.3f' % (
        roc_auc_score(y == 2, tr.n_blobs), roc_auc_score(y[y > 0] == 2, tr.n_blobs[y > 0]), roc_auc_score(y == 0, tr.n_blobs)))
    print('Contrast kept at blob peaks (mean over images):', d[keep_cols].mean().round(3).to_dict())
    print('Images whose blobs keep <50% contrast at 1520x912:', int((d.keep_1520x912 < .5).sum()), '/', int(d.keep_1520x912.notna().sum()))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    d[keep_cols].plot.box(ax=axes[0]); axes[0].set_title('top-hat contrast kept at blob peaks')
    for lab in CLASSES:
        axes[1].hist(tr[tr.label == lab].n_blobs.clip(upper=80), bins=20, alpha=.5, label=lab)
    axes[1].legend(); axes[1].set_title('small bright blobs per image (clipped 80)')
    fig.tight_layout(); fig.savefig(out / 'calc_resolution.png', dpi=110); plt.close(fig)
    ims, titles = [], []
    for lab in CLASSES:
        for iid in tr[tr.label == lab].sort_values('n_blobs', ascending=False).image_id[:2]:
            crop, peaks = crops[iid]
            cy, cx = peaks[len(peaks) // 2]
            y0, x0 = max(0, cy - 96), max(0, cx - 96)
            patch = crop[y0:y0 + 192, x0:x0 + 192]
            ims.append(patch); titles.append(f'{iid[6:10]} {lab} native/2')
            for name, (h, w) in SIZES.items():
                small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
                back = cv2.resize(small, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
                ims.append(back[y0:y0 + 192, x0:x0 + 192]); titles.append(name)
    grid(ims, titles, out / 'calc_zoom_per_class.png', cols=8)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
