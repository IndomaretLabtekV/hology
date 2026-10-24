"""Stacking test: v6 OOF probabilities + breast-density proxies (tissue_p50, dense_frac from EDA 37) -> logistic regression.
Scored on train with nested group folds (honest); the post-competition test check uses ref/ground_truth.csv for diagnosis only."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from deep_common import CLASSES, ROOT, output, plt

RUNS = ['v5_control', 'robust_rot3', 'robust_rot3_s1']
DENS = ['tissue_p50', 'dense_frac']


def main():
    out = output(38)
    lp = pd.read_csv(ROOT / 'outputs/37/lesion_proxy.csv').set_index('image_id')
    folds = pd.read_csv(ROOT / 'outputs/19/fold_manifest.csv').set_index('image_id')
    gt = pd.read_csv(ROOT / 'ref/ground_truth.csv').set_index('image_id').label
    oof = {r: pd.read_csv(ROOT / f'results/v6/oof_{r}.csv').set_index('image_id') for r in RUNS}
    tst = {r: pd.read_csv(ROOT / f'results/v6/test_{r}.csv').set_index('image_id').reindex(gt.index) for r in RUNS}
    ids = oof[RUNS[0]].index
    y = oof[RUNS[0]].label.map(CLASSES.index).to_numpy(); yt = gt.map(CLASSES.index).to_numpy()
    fold = folds.loc[ids, 'fold_group_union'].to_numpy()
    pcols = [f'p_{c}' for c in CLASSES]
    rows = []
    for name, runs in [('s1', ['robust_rot3_s1']), ('rot3', ['robust_rot3']), ('ens3', RUNS)]:
        P = np.mean([oof[r][pcols].to_numpy() for r in runs], 0); PT = np.mean([tst[r][pcols].to_numpy() for r in runs], 0)
        for feats in ['probs', 'probs+density']:
            X = np.log(np.clip(P, 1e-6, 1)); XT = np.log(np.clip(PT, 1e-6, 1))
            if feats == 'probs+density':
                X = np.hstack([X, lp.loc[ids, DENS].to_numpy()]); XT = np.hstack([XT, lp.loc[gt.index, DENS].to_numpy()])
            for C in [0.1, 1.0]:
                pred = np.zeros(len(y), int)
                for f in np.unique(fold):
                    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=3000))
                    pred[fold == f] = m.fit(X[fold != f], y[fold != f]).predict(X[fold == f])
                m = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=3000)).fit(X, y)
                rows.append(dict(base=name, feats=feats, C=C, oof_nested=f1_score(y, pred, average='macro'),
                                 base_oof=f1_score(y, P.argmax(1), average='macro'),
                                 test_diag=f1_score(yt, m.predict(XT), average='macro'), base_test_diag=f1_score(yt, PT.argmax(1), average='macro')))
    t = pd.DataFrame(rows).round(4); t.to_csv(out / 'density_stacking.csv', index=False)
    print(t.to_string(index=False))
    fig, ax = plt.subplots(figsize=(8, 4))
    t.assign(k=t.base + '/' + t.feats + '/C' + t.C.astype(str)).set_index('k')[['oof_nested', 'test_diag']].plot.barh(ax=ax)
    ax.axvline(t.base_oof.max(), ls='--', c='gray'); ax.set_title('stacking: nested OOF vs test (diagnosis)')
    fig.tight_layout(); fig.savefig(out / 'density_stacking.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
