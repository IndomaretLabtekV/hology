"""Mammo-FM input is min-max per image. With background = 0 the scale is set by the single brightest pixel.
If that pixel is a tiny artefact (edge strip, marker, calcification) the tissue is squeezed dark. Measure it per class,
then compare fixes: raw min-max (v5), tissue-percentile window, trim-before-mask."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deep_common import CLASSES, grid, manifest, output, plt
from prep_v6 import preprocess
from prep_v7 import fm_view


def minmax(a):
    f = a.astype(np.float32)
    return (f - f.min()) / max(f.max() - f.min(), 1)


def window(a, hi_pct=99.5):
    # tissue-percentile window: background stays 0, tissue median/contrast no longer hostage to one bright pixel
    t = a[a > 0]
    lo, hi = np.percentile(t, [0.5, hi_pct])
    w = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1) * 254 + 1
    return np.where(a > 0, w, 0).astype(np.uint8)


def robust_scale(a, pct=99.8, border=8):
    # keep 0 as black point (relative density preserved), take the white point from tissue away from the view border
    inner = np.zeros(a.shape, bool); inner[border:-border, border:-border] = True
    t = a[(a > 0) & inner]
    hi = np.percentile(t, pct) if t.size else 255
    return np.where(a > 0, np.clip(a.astype(np.float32) / max(hi, 1) * 255, 1, 255), 0).astype(np.uint8)


def saturation(a):
    t = a[a > 0]
    return float((t >= 254).mean())


def stats(r):
    img, mask, support, _ = preprocess(r.path, (2048, 1536), 'soft', True)
    fm = cv2.resize(fm_view(img, support), (456, 760), interpolation=cv2.INTER_AREA)
    t = fm[fm > 0]
    ys, xs = np.unravel_index(np.argmax(fm), fm.shape)
    edge = min(ys, xs, fm.shape[0] - 1 - ys, fm.shape[1] - 1 - xs) < 8
    top = (fm >= fm.max() - 5).sum()
    x = minmax(fm)[fm > 0]
    return dict(image_id=r.image_id, label=r.label, split=r.split, max=int(fm.max()), max_at_edge=bool(edge), n_near_max=int(top),
                p50=float(np.percentile(t, 50)), p99=float(np.percentile(t, 99)), p999=float(np.percentile(t, 99.9)),
                headroom=float(fm.max() - np.percentile(t, 99.9)), minmax_p50=float(np.median(x)),
                window_p50=float(np.median(window(fm)[fm > 0]) / 255), robust_p50=float(np.median(robust_scale(fm)[fm > 0]) / 255),
                sat_minmax=saturation((minmax(fm) * 255).astype(np.uint8)), sat_window=saturation(window(fm)), sat_robust=saturation(robust_scale(fm))), fm


def main():
    out = output(31)
    data = manifest()
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(stats, data.itertuples(index=False)))
    d = pd.DataFrame([r[0] for r in res]); views = {r[0]['image_id']: r[1] for r in res}
    d.to_csv(out / 'intensity_stats.csv', index=False)
    cols = ['max', 'p50', 'p99', 'p999', 'headroom', 'minmax_p50', 'window_p50', 'robust_p50', 'sat_minmax', 'sat_window', 'sat_robust', 'n_near_max']
    print(d.groupby(['split', 'label'])[cols].median().round(3).to_string())
    print('\nmax pixel on the view border:', d.groupby('label').max_at_edge.mean().round(3).to_dict())
    tr = d[d.split == 'train']; y = tr.label.map(CLASSES.index)
    rows = []
    for c in cols:
        rows.append(dict(feature=c, auc_normal=roc_auc_score(y == 0, tr[c]), auc_malignant_vs_rest=roc_auc_score(y == 2, tr[c]),
                         auc_m_vs_b=roc_auc_score(y[y > 0] == 2, tr[c][y > 0])))
    auc = pd.DataFrame(rows).round(3); auc.to_csv(out / 'intensity_auc.csv', index=False)
    print('\nUnivariate AUC (0.5 = no signal):\n', auc.to_string(index=False))
    dark = d.sort_values('minmax_p50').head(12)
    print('\nDarkest tissue after v5 min-max:\n', dark[['image_id', 'label', 'max', 'p50', 'p999', 'headroom', 'minmax_p50', 'window_p50']].round(3).to_string(index=False))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, c in zip(axes, ['minmax_p50', 'window_p50', 'robust_p50']):
        for lab in CLASSES + ['?']:
            ax.hist(d[d.label == lab][c], bins=25, alpha=.45, label=lab, density=True)
        ax.set_title(c); ax.legend()
    fig.tight_layout(); fig.savefig(out / 'intensity_hist.png', dpi=110); plt.close(fig)
    picks = list(dark.image_id[:4]) + ['image_0087.jpg', 'image_0159.jpg']
    for c in CLASSES:
        picks += data[data.label == c].sample(2, random_state=5).image_id.tolist()
    ims, titles = [], []
    for iid in picks:
        v = views[iid]; lab = d.set_index('image_id').label[iid]
        ims += [(minmax(v) * 255).astype(np.uint8), window(v), robust_scale(v)]
        titles += [f'{iid[6:10]} {lab} v5 min-max', 'window p0.5-99.5', 'robust max (0..p99.8 inner)']
    grid(ims, titles, out / 'window_per_class.png', cols=6)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
