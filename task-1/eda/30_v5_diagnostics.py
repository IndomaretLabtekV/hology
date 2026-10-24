"""v5 post-mortem without test labels: OOF confusion (from results/v5/confusion_best.png), fold variance,
fusion-weight curve, agreement matrix between candidate submissions, CC/MLO pair consistency on test."""
import itertools

import numpy as np
import pandas as pd

from deep_common import CLASSES, ROOT, output, plt

# Read off results/v5/confusion_best.png (OOF ft_fm, per image, rows = true N/B/M). v5 did not save OOF probabilities.
CONF = np.array([[70, 6, 4], [14, 31, 7], [19, 19, 42]])


def main():
    out = output(30)
    v5 = ROOT / 'results/v5'
    tp = np.diag(CONF); prec = tp / CONF.sum(0); rec = tp / CONF.sum(1); f1 = 2 * prec * rec / (prec + rec)
    print(pd.DataFrame({'precision': prec, 'recall': rec, 'f1': f1, 'true_n': CONF.sum(1), 'pred_n': CONF.sum(0)}, index=CLASSES).round(3))
    print('macro F1 from confusion:', round(f1.mean(), 4))
    # what-if: move k Malignant->Normal errors to correct, how much macro F1 is on the table
    for k in [5, 10, 19]:
        c = CONF.copy(); c[2, 0] -= k; c[2, 2] += k
        t = np.diag(c); p = t / c.sum(0); r = t / c.sum(1)
        print(f'fix {k:2d} M->N errors: macro F1 {np.mean(2 * p * r / (p + r)):.3f}')

    hist = pd.concat([pd.read_csv(v5 / f'history_fm_{f}.csv').assign(fold=f) for f in range(5)], ignore_index=True)
    last = hist[hist.epoch == 12].val_f1
    print(f'\nFold val F1 at last epoch: {last.round(3).tolist()}  mean {last.mean():.3f} sd {last.std():.3f}')
    print('Best epoch per fold:', hist.loc[hist.groupby('fold').val_f1.idxmax()][['fold', 'epoch', 'val_f1']].round(3).values.tolist())

    comp = pd.read_csv(v5 / 'model_comparison.csv')
    fus = comp[comp.model.str.startswith('fusion')].assign(w=lambda d: d.model.str[8:].astype(float))
    print('\nFusion weight curve (w = Mammo-FM share):\n', fus.pivot(index='w', columns='pair_fusion', values='macro_f1').round(3).to_string())

    subs = {p.stem.replace('submission_', ''): pd.read_csv(p).set_index('image_id').label for p in sorted(v5.glob('submission_*.csv'))}
    keys = ['ft_fm', 'ft_fm_pair', 'probe_fm', 'probe_dino', 'probe_dino_crop', 'probe_concat', 'fusion_w0.5', 'fusion_w0.75']
    agree = pd.DataFrame([[(subs[a] == subs[b]).mean() for b in keys] for a in keys], index=keys, columns=keys)
    print('\nTest agreement matrix (fraction same label):\n', agree.round(2).to_string())
    dist = pd.DataFrame({k: subs[k].value_counts().reindex(CLASSES).fillna(0).astype(int) for k in keys}).T
    print('\nTest predicted counts (dataset totals minus train imply ~20/14/20, used only as sanity):\n', dist.to_string())

    tags = pd.read_csv(ROOT / 'outputs/13/pairs.csv').set_index('image_id')
    te = tags[tags.split == 'test']
    paired = te[te.duplicated('pair', keep=False)]
    for k in ['ft_fm', 'probe_fm', 'probe_dino']:
        s = subs[k].reindex(paired.index)
        consistent = s.groupby(paired.pair).nunique().eq(1).mean()
        print(f'{k}: CC/MLO pairs with the same predicted label: {consistent:.2f} ({paired.pair.nunique()} pairs, ConvNeXt pairing ~0.79 precision)')
    ft = subs['ft_fm'].reindex(paired.index)
    split_pairs = ft.groupby(paired.pair).agg(lambda s: '/'.join(sorted(s))).value_counts()
    print('ft_fm label combos per test pair:\n', split_pairs.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].imshow(CONF, cmap='Blues')
    for i, j in itertools.product(range(3), range(3)):
        axes[0].text(j, i, CONF[i, j], ha='center', va='center')
    axes[0].set_xticks(range(3), CLASSES); axes[0].set_yticks(range(3), CLASSES); axes[0].set_title('v5 OOF confusion (true rows)')
    for f, g in hist.groupby('fold'):
        axes[1].plot(g.epoch, g.val_f1, label=f'fold {f}')
    axes[1].legend(); axes[1].set_title('val macro F1 per epoch')
    im = axes[2].imshow(agree, vmin=0.3, vmax=1, cmap='viridis'); axes[2].set_xticks(range(len(keys)), keys, rotation=60, fontsize=7)
    axes[2].set_yticks(range(len(keys)), keys, fontsize=7); axes[2].set_title('test agreement between submissions'); fig.colorbar(im, ax=axes[2])
    fig.tight_layout(); fig.savefig(out / 'v5_diagnostics.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
