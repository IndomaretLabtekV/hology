"""Audit pairing without using labels to choose links. Reuses existing embeddings, no forward pass."""
import json

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedGroupKFold

from deep_common import ROOT, CLASSES, grid, manifest, output, plt
from prep_v6 import read_gray


def cosine(x):
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)) @ (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)).T


def matched_edges(data, similarity):
    edges = []
    for _, cell in data.groupby('lat2'):
        cc = cell.index[cell.view2 == 'CC'].to_numpy()
        ml = cell.index[cell.view2 == 'MLO'].to_numpy()
        if not len(cc) or not len(ml):
            continue
        a, b = linear_sum_assignment(-similarity[np.ix_(cc, ml)])
        for i, j in zip(cc[a], ml[b]):
            alternatives = similarity[i, ml[ml != j]]
            margin = similarity[i, j] - alternatives.max() if len(alternatives) else 1.0
            edges.append((int(i), int(j), float(similarity[i, j]), float(margin)))
    return edges


def components(n, edges):
    parent = np.arange(n)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in edges:
        a, b = root(i), root(j)
        parent[max(a, b)] = min(a, b)
    return np.array([root(i) for i in range(n)])


def main():
    out = output(19)
    data = manifest()
    tags = pd.read_csv(ROOT / 'outputs/13/pairs.csv')
    tr = data[data.split == 'train'].merge(tags[['image_id', 'lat2', 'view2']], on='image_id', validate='one_to_one')
    order_all = pd.read_csv(ROOT / 'outputs/05/groups.csv')
    order = order_all.loc[order_all.label != '?', 'image_id']
    fm = pd.DataFrame(np.load(ROOT / 'outputs/12/feat_mammo_fm_fm_notext_s2_h.npy'), index=order).loc[tr.image_id].to_numpy()
    conv = pd.DataFrame(np.load(ROOT / 'outputs/05/emb_convnext.npy'), index=order_all.image_id).loc[tr.image_id].to_numpy()
    # Scaling here is exclusively for unsupervised group reconstruction, never classifier normalization.
    fm = (fm - fm.mean(0)) / np.maximum(fm.std(0), 1e-6)
    sf, sc = cosine(fm), cosine(conv)
    mixed = (sf - sf.mean()) / sf.std() + (sc - sc.mean()) / sc.std()
    first, second = matched_edges(tr, sf), matched_edges(tr, mixed)
    first_set, second_set = {tuple(sorted((a, b))) for a, b, _, _ in first}, {tuple(sorted((a, b))) for a, b, _, _ in second}
    union = first_set | second_set
    # Exact decoded duplicates have stronger evidence than embedding similarity.
    integrity = pd.read_csv(ROOT / 'outputs/18/integrity.csv')
    merged = tr[['image_id']].merge(integrity[['image_id', 'pixel_sha256']], on='image_id', validate='one_to_one')
    for _, d in merged.groupby('pixel_sha256'):
        union.update((int(d.index[0]), int(i)) for i in d.index[1:])
    group = components(len(tr), union)
    pair_group = components(len(tr), first_set)
    tr['group_union'], tr['group_single'] = group, pair_group
    y = tr.label.map(CLASSES.index).to_numpy()
    fold_rows = []
    for key in ['group_union', 'group_single']:
        tr['fold_' + key] = -1
        for f, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(tr, y, tr[key])):
            assert not set(tr[key].iloc[a]) & set(tr[key].iloc[b])
            tr.loc[b, 'fold_' + key] = f
            fold_rows.append(dict(grouping=key, fold=f, images=len(b), groups=tr[key].iloc[b].nunique(),
                                  **{c: int((y[b] == k).sum()) for k, c in enumerate(CLASSES)}))
    tr.drop(columns='path').to_csv(out / 'fold_manifest.csv', index=False)
    records = []
    for name, edges in [('fm', first), ('fm_conv', second)]:
        for a, b, score, margin in edges:
            records.append(dict(method=name, a=tr.image_id[a], b=tr.image_id[b], score=score, margin=margin,
                                stable=tuple(sorted((a, b))) in first_set & second_set,
                                same_label=bool(y[a] == y[b])))
    links = pd.DataFrame(records)
    links.to_csv(out / 'candidate_pairs.csv', index=False)
    counts = tr.groupby('group_union').size()
    summary = dict(pair_count=len(first_set), stable_pairs=len(first_set & second_set), union_edges=len(union),
                   union_groups=len(counts), max_union_group=int(counts.max()),
                   same_label_rate=links.groupby('method').same_label.mean().to_dict(),
                   warning='same_label_rate is NOT patient matching accuracy; unknown OCR/pairing errors remain. Groups use cached Mammo-FM/ConvNeXt features, not DINOv3 features.')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)); print(pd.DataFrame(fold_rows).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(counts, bins=np.arange(0.5, counts.max() + 1.5)); axes[0].set_title('Conservative component sizes')
    for name, d in links.groupby('method'):
        axes[1].hist(d.margin, bins=20, alpha=0.5, label=name)
    axes[1].axvline(0, color='black'); axes[1].legend(); axes[1].set_title('Assigned minus best alternative similarity')
    fig.tight_layout(); fig.savefig(out / 'pair_uncertainty.png', dpi=120); plt.close(fig)
    selected = links[links.method == 'fm'].sort_values(['stable', 'margin']).head(12)
    images, titles = [], []
    paths = data.set_index('image_id').path
    for r in selected.itertuples():
        for iid in [r.a, r.b]:
            images.append(read_gray(paths[iid], scale=8)); titles.append(f'{iid}\nstable={r.stable}, margin={r.margin:.3f}')
    grid(images, titles, out / 'ambiguous_train_pairs.png', cols=6)
    # Near-duplicate candidates are reviewed, not automatically collapsed into patients.
    thumbs = []
    for row in data.itertuples():
        a = cv2.resize(read_gray(row.path, 8), (96, 128)).astype(np.float32)
        a -= a.mean(); thumbs.append(a)
    flat = np.stack(thumbs).reshape(len(data), -1)
    similarity = cosine(flat)
    mirrored = np.stack([a[:, ::-1] for a in thumbs]).reshape(len(data), -1)
    normal = flat / np.linalg.norm(flat, axis=1, keepdims=True)
    flip = mirrored / np.linalg.norm(mirrored, axis=1, keepdims=True)
    similarity = np.maximum(similarity, normal @ flip.T)
    a, b = np.triu_indices(len(data), 1)
    ranked = np.argsort(similarity[a, b])[-24:][::-1]
    near = pd.DataFrame([dict(a=data.image_id[a[k]], b=data.image_id[b[k]], correlation=similarity[a[k], b[k]],
                              cross_split=bool(data.split[a[k]] != data.split[b[k]])) for k in ranked])
    near.to_csv(out / 'near_duplicate_candidates.csv', index=False)
    print('Top thumbnail similarities, NOT confirmed duplicates:\n', near.head(8).to_string(index=False))
    grid([read_gray(paths[i], 8) for r in near.head(6).itertuples() for i in [r.a, r.b]],
         [f'{i}\nr={r.correlation:.4f} cross={r.cross_split}' for r in near.head(6).itertuples() for i in [r.a, r.b]],
         out / 'near_duplicate_candidates.png')


if __name__ == '__main__':
    main()
