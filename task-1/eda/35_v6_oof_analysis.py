"""v6 OOF deep-dive (probabilities per image, no model run):
1) stability across the 3 runs (seed/prep noise), always-wrong images
2) CC/MLO fusion rules: mean (v5/v6) vs max-abnormal ("a lesion seen in one view is enough") vs confidence-weighted
3) class bias / calibration per rule, nested over the same folds."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

from deep_common import CLASSES, ROOT, output, plt

RUNS = ['v5_control', 'robust_rot3', 'robust_rot3_s1']


def load():
    v6 = ROOT / 'results/v6'
    base = pd.read_csv(v6 / f'oof_{RUNS[0]}.csv')[['image_id', 'label']]
    P = {r: pd.read_csv(v6 / f'oof_{r}.csv')[[f'p_{c}' for c in CLASSES]].to_numpy() for r in RUNS}
    folds = pd.read_csv(ROOT / 'outputs/19/fold_manifest.csv').set_index('image_id')
    pairs = pd.read_csv(ROOT / 'outputs/16/train_pairs_fm.csv').set_index('image_id')
    base['fold'] = folds.loc[base.image_id, 'fold_group_union'].to_numpy()
    base['pair'] = pairs.loc[base.image_id, 'pair_fm'].to_numpy()
    return base, P


def fuse(p, pair, rule):
    g = pd.DataFrame(p).groupby(pair)
    if rule == 'none':
        return p
    if rule == 'mean':
        return g.transform('mean').to_numpy()
    if rule == 'max_abnormal':
        # abnormal classes take the stronger view, Normal the weaker; renormalise
        q = \
            np.column_stack([g[0].transform('min'), g[1].transform('max'), g[2].transform('max')])
        return q / q.sum(1, keepdims=True)
    if rule == 'confident':
        conf = pd.Series(p.max(1)).groupby(pair).transform('max').to_numpy()
        pick = pd.Series(np.where(p.max(1) == conf, np.arange(len(p)), -1)).groupby(pair).transform('max').to_numpy()
        return p[pick]
    raise ValueError(rule)


def nested_bias(p, y, folds, grid=np.round(np.linspace(-1.5, 1.5, 31), 2)):
    lp = np.log(np.clip(p, 1e-8, 1)); out = np.zeros(len(y), int)
    for f in np.unique(folds):
        a, b = folds != f, folds == f
        best, s0 = (0, 0), -1
        for bb in grid:
            for bm in grid:
                s = f1_score(y[a], (lp[a] + [0, bb, bm]).argmax(1), average='macro')
                if s > s0 + 1e-9:
                    best, s0 = (bb, bm), s
        out[b] = (lp[b] + [0, *best]).argmax(1)
    return f1_score(y, out, average='macro')


def main():
    out = output(35)
    d, P = load()
    y = d.label.map(CLASSES.index).to_numpy()
    preds = np.stack([P[r].argmax(1) for r in RUNS])
    wrong = preds != y
    print('Per-run OOF macro F1:', {r: round(f1_score(y, P[r].argmax(1), average='macro'), 4) for r in RUNS})
    print('Prediction flips between runs (fraction of images where runs disagree):', round((preds.min(0) != preds.max(0)).mean(), 3))
    print('Always wrong (3/3):', int(wrong.all(0).sum()), '| sometimes wrong:', int((wrong.any(0) & ~wrong.all(0)).sum()),
          '| always right:', int((~wrong).all(0).sum()))
    always = d[wrong.all(0)].assign(pred=[CLASSES[k] for k in np.mean([P[r] for r in RUNS], 0)[wrong.all(0)].argmax(1)])
    print('Always-wrong by true->pred:\n', always.groupby(['label', 'pred']).size().to_string())
    always.to_csv(out / 'always_wrong.csv', index=False)

    ens = np.mean([P[r] for r in RUNS], 0)
    rows = []
    for name, p in [(r, P[r]) for r in RUNS] + [('ens3', ens), ('ens_robust', (P['robust_rot3'] + P['robust_rot3_s1']) / 2)]:
        for rule in ['none', 'mean', 'max_abnormal', 'confident']:
            q = fuse(p, d.pair.to_numpy(), rule)
            rows.append(dict(model=name, rule=rule, macro_f1=f1_score(y, q.argmax(1), average='macro'),
                             bias_nested=nested_bias(q, y, d.fold.to_numpy()), pred_malignant=int((q.argmax(1) == 2).sum()),
                             recall_malignant=float((q.argmax(1)[y == 2] == 2).mean())))
    t = pd.DataFrame(rows).round(4)
    t.to_csv(out / 'fusion_rules.csv', index=False)
    print('\n', t.pivot(index='model', columns='rule', values='macro_f1').to_string())
    print('\nwith nested bias:\n', t.pivot(index='model', columns='rule', values='bias_nested').to_string())
    print('\nMalignant recall:\n', t.pivot(index='model', columns='rule', values='recall_malignant').to_string())
    print('\nMean over models per rule:', t.groupby('rule')[['macro_f1', 'bias_nested']].mean().round(4).to_dict())

    q = fuse(ens, d.pair.to_numpy(), 'mean')
    same = pd.Series(y).groupby(d.pair.to_numpy()).transform('nunique').to_numpy() == 1
    disagree = pd.Series(ens.argmax(1)).groupby(d.pair.to_numpy()).transform('nunique').to_numpy() > 1
    print(f'\nPairs with disagreeing view predictions: {disagree.mean():.2f} of images; error rate there '
          f'{(ens.argmax(1) != y)[disagree].mean():.2f} vs {(ens.argmax(1) != y)[~disagree].mean():.2f}; label-consistent pairs {same.mean():.2f}')
    print('Confusion ens3 + mean fusion:\n', confusion_matrix(y, q.argmax(1)))
    conf = ens.max(1)
    bins = pd.cut(conf, [0, .5, .6, .7, .8, 1])
    print('Reliability (confidence bin -> accuracy, n):\n', pd.DataFrame({'acc': ens.argmax(1) == y, 'bin': bins}).groupby('bin', observed=True).acc.agg(['mean', 'size']).round(3).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    t.pivot(index='model', columns='rule', values='macro_f1').plot.bar(ax=axes[0]); axes[0].set_ylim(.55, .72); axes[0].set_title('OOF macro F1 by pair-fusion rule')
    axes[1].hist([ens[y == k, 2] for k in range(3)], bins=20, label=CLASSES, stacked=False, alpha=.6); axes[1].legend(); axes[1].set_title('ens3 p(Malignant) by true class')
    fig.tight_layout(); fig.savefig(out / 'fusion_rules.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
