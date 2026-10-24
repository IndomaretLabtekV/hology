"""Duplication / memorization audit on cached features (no new forward pass).
1) How much does ignoring the CC+MLO pairing inflate CV (1-NN and LR probe)?
2) Do union groups mix labels (pairing errors)?
3) Does any test image look like a train partner (cross-split duplicate)?"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deep_common import CLASSES, ROOT, output, plt


def cos(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True); b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a @ b.T


def oof(X, y, folds, kind):
    p = np.zeros(len(y), int)
    for a, b in folds:
        if kind == '1nn':
            s = cos(X[b], X[a]); p[b] = y[a][s.argmax(1)]
        else:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight='balanced', max_iter=3000))
            p[b] = m.fit(X[a], y[a]).predict(X[b])
    return f1_score(y, p, average='macro')


def main():
    out = output(28)
    man = pd.read_csv(ROOT / 'outputs/19/fold_manifest.csv')
    order = pd.read_csv(ROOT / 'outputs/05/groups.csv')
    tr_order = order.loc[order.label != '?', 'image_id']
    fm = pd.DataFrame(np.load(ROOT / 'outputs/12/feat_mammo_fm_fm_notext_s2_h.npy'), index=tr_order).loc[man.image_id].to_numpy()
    y = man.label.map(CLASSES.index).to_numpy()
    pairs = pd.read_csv(ROOT / 'outputs/16/train_pairs_fm.csv').set_index('image_id').loc[man.image_id, 'pair_fm'].to_numpy()
    grouped = [(np.flatnonzero(man.fold_group_union != f), np.flatnonzero(man.fold_group_union == f)) for f in range(5)]
    rows = []
    for seed in range(5):
        naive = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(fm, y))
        for kind in ['1nn', 'lr']:
            rows.append(dict(seed=seed, model=kind, cv='naive_stratified', macro_f1=oof(fm, y, naive, kind)))
    for kind in ['1nn', 'lr']:
        rows.append(dict(seed=0, model=kind, cv='group_union', macro_f1=oof(fm, y, grouped, kind)))
    d = pd.DataFrame(rows)
    d.to_csv(out / 'cv_inflation.csv', index=False)
    print(d.groupby(['model', 'cv']).macro_f1.agg(['mean', 'std']).round(3).to_string())
    s = cos(fm, fm); np.fill_diagonal(s, -1)
    nn = s.argmax(1)
    partner_hit = (pairs[nn] == pairs).mean()
    print(f'\nTop-1 neighbour is the same patient pair: {partner_hit:.3f} (chance ~{1 / 211:.3f})')
    print(f'Top-1 neighbour same label: {(y[nn] == y).mean():.3f}')
    mixed = man.groupby('group_union').label.nunique()
    print(f'Union groups: {len(mixed)}, groups with >1 label: {(mixed > 1).sum()} '
          f'({man.group_union.isin(mixed[mixed > 1].index).sum()} images)')
    print('Fold class counts:\n', pd.crosstab(man.fold_group_union, man.label).to_string())
    emb = np.load(ROOT / 'outputs/05/emb_convnext.npy')
    split = np.where(order.label == '?', 'test', 'train')
    sc = cos(emb, emb); np.fill_diagonal(sc, -1)
    tr_idx, te_idx = np.flatnonzero(split == 'train'), np.flatnonzero(split == 'test')
    pos = order.set_index('image_id').index
    pair_of = dict(zip(man.image_id, pairs))
    tri = [(i, j) for i in tr_idx for j in tr_idx if i < j]
    same = np.array([pair_of[pos[i]] == pair_of[pos[j]] for i, j in tri])
    vals = np.array([sc[i, j] for i, j in tri])
    cross = sc[np.ix_(te_idx, tr_idx)].max(1)
    thr = np.percentile(vals[same], 50)
    print(f'\nConvNeXt cosine: train true-pair median {np.median(vals[same]):.3f}, train non-pair p99 {np.percentile(vals[~same], 99):.3f}, '
          f'test->train max median {np.median(cross):.3f}; test images above pair median: {(cross > thr).sum()}/54')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    d.groupby(['model', 'cv']).macro_f1.mean().unstack().plot.bar(ax=axes[0]); axes[0].set_title('Macro F1: naive vs grouped CV')
    axes[1].hist(vals[~same], bins=60, alpha=.5, density=True, label='train non-pair')
    axes[1].hist(vals[same], bins=30, alpha=.5, density=True, label='train same pair')
    axes[1].hist(cross, bins=20, alpha=.5, density=True, label='test -> best train')
    axes[1].legend(); axes[1].set_title('ConvNeXt cosine similarity')
    fig.tight_layout(); fig.savefig(out / 'memorization.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
