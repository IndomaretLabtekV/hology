"""Post-competition audit: every saved submission / test-probability file vs ref/ground_truth.csv.
Diagnosis only (late submission). Scores, confusion, per-image error frequency, and what post-processing
(pair fusion rule, ensemble, class bias) would have changed on the v6 test probabilities."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from deep_common import CLASSES, ROOT, output, plt


def main():
    out = output(36)
    gt = pd.read_csv(ROOT / 'ref/ground_truth.csv').set_index('image_id').label
    print('GT counts:', gt.value_counts().to_dict())
    rows, wrong = [], {}
    for p in sorted(Path(ROOT / 'results').rglob('submission*.csv')):
        s = pd.read_csv(p).set_index('image_id').label.reindex(gt.index)
        if s.isna().any():
            continue
        name = str(p.relative_to(ROOT / 'results'))
        rows.append(dict(file=name, macro_f1=f1_score(gt, s, average='macro'), acc=(s == gt).mean(),
                         **{f'f1_{c}': f1_score(gt == c, s == c) for c in CLASSES}))
        wrong[name] = s != gt
    t = pd.DataFrame(rows).sort_values('macro_f1', ascending=False).round(4)
    t.to_csv(out / 'submission_scores.csv', index=False)
    print(t.head(25).to_string(index=False))
    W = pd.DataFrame(wrong)
    freq = W.mean(1).sort_values(ascending=False)
    print('\nImages wrong in >=80% of submissions:\n', pd.DataFrame({'gt': gt, 'wrong_rate': freq}).query('wrong_rate >= .8').to_string())

    v6 = ROOT / 'results/v6'
    P = {r: pd.read_csv(v6 / f'test_{r}.csv').set_index('image_id').reindex(gt.index)[[f'p_{c}' for c in CLASSES]].to_numpy()
         for r in ['v5_control', 'robust_rot3', 'robust_rot3_s1']}
    y = gt.map(CLASSES.index).to_numpy()
    tags = pd.read_csv(ROOT / 'outputs/13/pairs.csv').set_index('image_id').reindex(gt.index)
    pair = tags.pair.to_numpy()

    def fuse(p, rule):
        g = pd.DataFrame(p).groupby(pair)
        if rule == 'none':
            return p
        if rule == 'mean':
            return g.transform('mean').to_numpy()
        q = np.column_stack([g[0].transform('min'), g[1].transform('max'), g[2].transform('max')])
        return q / q.sum(1, keepdims=True)

    same = pd.Series(y).groupby(pair).transform('nunique').eq(1).mean()
    print(f'\nTest pairs (ConvNeXt/OCR pairing) label-consistent: {same:.2f}')
    ens = {'ens3': np.mean(list(P.values()), 0), 'ens_robust': (P['robust_rot3'] + P['robust_rot3_s1']) / 2, **P}
    fr = []
    for name, p in ens.items():
        for rule in ['none', 'mean', 'max_abnormal']:
            q = fuse(p, rule)
            fr.append(dict(model=name, rule=rule, macro_f1=f1_score(y, q.argmax(1), average='macro')))
    fr = pd.DataFrame(fr).pivot(index='model', columns='rule', values='macro_f1').round(4)
    print('\nv6 test macro F1 by fusion rule:\n', fr.to_string())
    q = fuse(ens['ens3'], 'mean')
    print('\nConfusion ens3/mean (rows true N/B/M):\n', confusion_matrix(y, q.argmax(1)))
    lp = np.log(np.clip(q, 1e-8, 1))
    grid = np.round(np.linspace(-1.5, 1.5, 31), 2)
    surf = np.array([[f1_score(y, (lp + [0, bb, bm]).argmax(1), average='macro') for bm in grid] for bb in grid])
    i, j = np.unravel_index(surf.argmax(), surf.shape)
    print(f'Bias surface on test (diagnostic ceiling): best {surf.max():.3f} at benign {grid[i]}, malignant {grid[j]}; no bias {surf[15, 15]:.3f}')
    per = pd.DataFrame({'gt': gt, 'pred': np.asarray(CLASSES)[q.argmax(1)], 'p_N': q[:, 0], 'p_B': q[:, 1], 'p_M': q[:, 2],
                        'view': tags.view2, 'pair': pair}).round(3)
    per[per.gt != per.pred].to_csv(out / 'v6_ens_errors.csv')
    print('\nv6 ens3/mean errors:\n', per[per.gt != per.pred].to_string())
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].imshow(surf, origin='lower', extent=[grid[0], grid[-1], grid[0], grid[-1]], cmap='viridis'); ax[0].set_xlabel('bias Malignant'); ax[0].set_ylabel('bias Benign')
    ax[0].set_title('test macro F1 vs class bias (diagnostic)')
    t.head(15).set_index('file').macro_f1.plot.barh(ax=ax[1]); ax[1].set_title('submission macro F1')
    fig.tight_layout(); fig.savefig(out / 'test_error_audit.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
