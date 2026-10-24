"""Chest-wall side audit: which canvas border does the breast touch? Left/right flip alone misses top/bottom-attached views."""
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deep_common import CLASSES, grid, manifest, output, plt
from prep import breast_mask
from prep_v6 import read_gray

SIDES = ['left', 'right', 'top', 'bottom']


def contacts(mask, band=0.03):
    h, w = mask.shape
    bh, bw = max(2, int(h * band)), max(2, int(w * band))
    # fraction of each border band covered by tissue, measured along that border
    return dict(left=mask[:, :bw].any(1).mean(), right=mask[:, -bw:].any(1).mean(),
                top=mask[:bh].any(0).mean(), bottom=mask[-bh:].any(0).mean())


def main():
    out = output(25)
    data = manifest()
    rows = []
    for r in data.itertuples():
        a = read_gray(r.path, 4, True)
        m = breast_mask(a)
        c = contacts(m)
        side = max(c, key=c.get)
        ys, xs = np.where(m)
        rows.append(dict(image_id=r.image_id, label=r.label, split=r.split, side=side, **{f'c_{k}': v for k, v in c.items()},
                         bbox_w=(xs.max() - xs.min() + 1) / m.shape[1], bbox_h=(ys.max() - ys.min() + 1) / m.shape[0],
                         area=m.mean()))
    d = pd.DataFrame(rows)
    d['vertical'] = d.side.isin(['top', 'bottom'])
    d.to_csv(out / 'orientation.csv', index=False)
    print(pd.crosstab([d.split, d.label], d.side, margins=True).to_string())
    tr = d[d.split == 'train']
    y = tr.label.map(CLASSES.index)
    print('\nAUC vertical-orientation vs Normal:', round(roc_auc_score(y == 0, tr.vertical), 3),
          '| vs Malignant among abnormal:', round(roc_auc_score(y[y > 0] == 2, tr.vertical[y > 0]), 3))
    for col in ['bbox_w', 'bbox_h', 'area']:
        print(f'{col}: AUC Normal {roc_auc_score(y == 0, tr[col]):.3f}  AUC M vs B {roc_auc_score(y[y > 0] == 2, tr[col][y > 0]):.3f}  '
              f'median by class {tr.groupby("label")[col].median().round(3).to_dict()}')
    amb = d[(d[[f'c_{s}' for s in SIDES]].apply(sorted, axis=1).str[-2] > 0.5 * d[[f'c_{s}' for s in SIDES]].max(1))]
    print('\nAmbiguous (second border >50% of best):', len(amb), amb.image_id.tolist()[:20])
    paths = data.set_index('image_id').path
    ims, titles = [], []
    for side in ['top', 'bottom', 'right', 'left']:
        for lab in CLASSES + ['?']:
            for iid in d[(d.side == side) & (d.label == lab)].image_id.head(2):
                ims.append(read_gray(paths[iid], 8)); titles.append(f'{iid[6:10]} {lab} {side}')
    grid(ims, titles, out / 'orientation_examples.png', cols=8)
    fig, ax = plt.subplots(figsize=(7, 4))
    pd.crosstab(d.label, d.side).plot.bar(ax=ax); ax.set_title('Chest-wall border by class')
    fig.tight_layout(); fig.savefig(out / 'orientation_by_class.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
