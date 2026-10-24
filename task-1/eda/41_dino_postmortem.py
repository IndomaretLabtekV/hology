"""Post-competition diagnosis with ref/ground_truth.csv: how did every DINOv3-based submission do (v4 friend, v5 probes)
compared to Mammo-FM (v5/v6), per class and per image? Where is DINOv3 right when Mammo-FM is wrong (complementarity)?"""
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from deep_common import CLASSES, ROOT, output, plt

FILES = {
    'v4_dino_vith_frozen_direct': 'v4/submission_dinov3_mlp.csv',
    'v4_dino_vith_frozen_modeprior': 'v4/submission_dinov3_mlp_best.csv',
    'v5_dino_vitb_probe_full': 'v5/submission_probe_dino.csv',
    'v5_dino_vitb_probe_crop': 'v5/submission_probe_dino_crop.csv',
    'v5_fm_probe': 'v5/submission_probe_fm.csv',
    'v5_fm_finetune': 'v5/submission_ft_fm.csv',
    'v6_fm_finetune_s1': 'v6/submission_robust_rot3_s1.csv',
}


def main():
    out = output(41)
    gt = pd.read_csv(ROOT / 'ref/ground_truth.csv').set_index('image_id').label
    S = pd.DataFrame({k: pd.read_csv(ROOT / 'results' / f).set_index('image_id').label.reindex(gt.index) for k, f in FILES.items()})
    rows = []
    for k in S:
        cm = confusion_matrix(gt, S[k], labels=CLASSES)
        rows.append(dict(model=k, macro_f1=f1_score(gt, S[k], average='macro'),
                         **{f'recall_{c}': cm[i, i] / cm[i].sum() for i, c in enumerate(CLASSES)},
                         pred_counts=S[k].value_counts().reindex(CLASSES).fillna(0).astype(int).tolist()))
    t = pd.DataFrame(rows).round(3); t.to_csv(out / 'dino_vs_fm.csv', index=False)
    print(t.to_string(index=False))
    ok = S.eq(gt, axis=0)
    fm, dino = ok['v6_fm_finetune_s1'], ok['v5_dino_vitb_probe_crop']
    print(f'\nFM right & DINO wrong: {(fm & ~dino).sum()} | DINO right & FM wrong: {(dino & ~fm).sum()} | both wrong: {(~fm & ~dino).sum()}')
    print('DINO-right/FM-wrong images:', gt[dino & ~fm].to_dict())
    print('Both wrong:', gt[~fm & ~dino].to_dict())
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(ok.astype(int).to_numpy(), aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    ax.set_xticks(range(len(S.columns)), S.columns, rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(len(gt)), [f'{i[6:10]} {l[0]}' for i, l in gt.items()], fontsize=6)
    ax.set_title('correct (green) per test image and model')
    fig.tight_layout(); fig.savefig(out / 'dino_vs_fm_grid.png', dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
