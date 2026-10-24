"""Lesion proxies on the v6 Mammo-FM view: largest bright compact region (big top-hat), its area, contrast, circularity,
and calcification-like dot count. Does any of them separate Benign vs Malignant in TRAIN, and where do the
post-competition test errors (ref/ground_truth.csv, diagnosis only) sit on those axes?"""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deep_common import CLASSES, ROOT, grid, manifest, output, plt
from prep_v6 import preprocess
from prep_v7 import fm_view, robust_scale

BIG = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))
SMALL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))


def features(r):
    img, mask, support, _ = preprocess(r.path, (2048, 1536), 'soft', True)
    v = cv2.resize(robust_scale(fm_view(img, support)), (456, 760), interpolation=cv2.INTER_AREA)
    inner = cv2.erode((v > 0).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    th = cv2.morphologyEx(cv2.GaussianBlur(v, (5, 5), 0), cv2.MORPH_TOPHAT, BIG).astype(np.float32)
    thr = max(25.0, float(np.percentile(th[inner], 99.5)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(((th > thr) & inner).astype(np.uint8), 8)
    if n > 1:
        k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
        blob = (lab == k).astype(np.uint8)
        cnt = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
        per = cv2.arcLength(cnt, True)
        area = float(st[k, cv2.CC_STAT_AREA])
        circ = 4 * np.pi * area / max(per ** 2, 1)
        contrast = float(th[blob > 0].mean())
        hull = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / max(hull, 1)
    else:
        area, circ, contrast, solidity = 0.0, np.nan, 0.0, np.nan
    dots = cv2.morphologyEx(v, cv2.MORPH_TOPHAT, SMALL).astype(np.float32)
    n_dots = int((cv2.connectedComponentsWithStats(((dots > 40) & inner).astype(np.uint8), 8)[0]) - 1)
    t = v[v > 0]
    return dict(image_id=r.image_id, label=r.label, split=r.split, blob_area=area, blob_circularity=circ, blob_solidity=solidity,
                blob_contrast=contrast, n_dots=n_dots, tissue_p50=float(np.median(t)), dense_frac=float((t > 180).mean())), v


def main():
    out = output(37)
    data = manifest()
    with ThreadPoolExecutor(6) as pool:
        res = list(pool.map(features, data.itertuples(index=False)))
    d = pd.DataFrame([r[0] for r in res]); views = {r[0]['image_id']: r[1] for r in res}
    gt = pd.read_csv(ROOT / 'ref/ground_truth.csv').set_index('image_id').label
    d['label_diag'] = d.label.where(d.split == 'train', d.image_id.map(gt))
    v6 = pd.read_csv(ROOT / 'results/v6/submission_robust_rot3_s1.csv').set_index('image_id').label
    d['v6_pred'] = d.image_id.map(v6)
    d.to_csv(out / 'lesion_proxy.csv', index=False)
    cols = ['blob_area', 'blob_circularity', 'blob_solidity', 'blob_contrast', 'n_dots', 'tissue_p50', 'dense_frac']
    tr = d[d.split == 'train']; y = tr.label.map(CLASSES.index)
    rows = []
    for c in cols:
        x = tr[c].fillna(tr[c].median())
        rows.append(dict(feature=c, auc_m_vs_b=roc_auc_score(y[y > 0] == 2, x[y > 0]), auc_abnormal=roc_auc_score(y > 0, x),
                         **{f'median_{k}': tr[tr.label == k][c].median() for k in CLASSES}))
    print('TRAIN univariate AUC:\n', pd.DataFrame(rows).round(3).to_string(index=False))
    te = d[d.split == 'test'].copy()
    te['case'] = np.where(te.label_diag == te.v6_pred, 'correct', te.label_diag + '->' + te.v6_pred)
    print('\nTEST (diagnosis) medians by v6 outcome:\n', te.groupby('case')[cols].median().round(3).to_string())
    print(te.case.value_counts().to_string())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (a, b) in zip(axes, [('blob_area', 'blob_contrast'), ('blob_circularity', 'blob_solidity'), ('n_dots', 'dense_frac')]):
        for lab, m in [('Benign', 'o'), ('Malignant', 's')]:
            s = tr[tr.label == lab]; ax.scatter(s[a], s[b], alpha=.4, marker=m, label=f'train {lab}')
        e = te[te.case == 'Malignant->Benign']; ax.scatter(e[a], e[b], c='red', marker='x', s=70, label='test M->B (v6)')
        ax.set_xlabel(a); ax.set_ylabel(b); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / 'lesion_proxy.png', dpi=110); plt.close(fig)
    ims, titles = [], []
    for lab in ['Benign', 'Malignant']:
        for iid in tr[tr.label == lab].sort_values('blob_contrast', ascending=False).image_id[:4]:
            ims.append(views[iid]); titles.append(f'{iid[6:10]} train {lab}')
    for iid in te[te.case == 'Malignant->Benign'].image_id[:6]:
        ims.append(views[iid]); titles.append(f'{iid[6:10]} test M->B')
    grid(ims, titles, out / 'lesion_examples.png', cols=7)


if __name__ == '__main__':
    cv2.setNumThreads(1)
    main()
