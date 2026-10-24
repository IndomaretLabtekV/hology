"""Run with .venv/bin/python eda/test_v7_pipeline.py. Static + unit checks only; no notebook cell is executed."""
import ast
import io
import json
import re

import cv2
import numpy as np
import pandas as pd

from deep_common import ROOT
from prep_v7 import augment_v7, cover_scale, fm_view, masked_jitter, robust_scale, rotate_cover, tissue_crop


def main():
    img = np.zeros((512, 384), np.uint8)
    cv2.ellipse(img, (0, 256), (230, 210), 0, 0, 360, 150, -1)
    for angle in [-7, -3, 0, 4, 7]:
        r = rotate_cover(np.full_like(img, 200), angle)
        assert (r > 0).all(), 'cover rotation left an empty corner'
        rows = (rotate_cover(img, angle) > 0).any(1)
        first = np.argmax(rotate_cover(img, angle) > 0, 1)
        assert (first[rows] <= 3).mean() > 0.95, 'chest wall detached after rotation'
    assert cover_scale(img.shape, 0) == 1.0
    j = masked_jitter(img, 1.1, 0.9)
    assert (j[img == 0] == 0).all() and (j[img > 0] > 0).all(), 'jitter touched background or zeroed tissue'
    a = augment_v7(img, np.random.default_rng(1)); b = augment_v7(img, np.random.default_rng(1))
    assert np.array_equal(a, b) and a.shape == img.shape and a.dtype == np.uint8
    art = img.copy(); art[:, :4][img[:, :4] > 0] = 255
    rs = robust_scale(art)
    assert (rs[img == 0] == 0).all() and np.median(rs[img > 0]) > 200, 'edge strip still sets the white point'
    assert (robust_scale(np.zeros_like(img)) == 0).all()
    tc = tissue_crop(img, img > 0)
    assert tc.shape[0] <= img.shape[0] and (tc[:, 0] > 0).mean() > 0.9, 'tissue crop lost the chest wall'
    v = fm_view(img, img > 0, (152, 91))
    assert v.shape == (152, 91) and (v[:, 0] > 0).mean() > 0.9
    try:
        fm_view(img, np.zeros_like(img, bool))
    except ValueError:
        pass
    else:
        raise AssertionError('empty support accepted')

    tags = pd.read_csv(ROOT / 'outputs/13/pairs.csv')
    assert len(tags) == 266 and tags[['lat2', 'view2']].notna().all().all() and set(tags.view2) == {'CC', 'MLO'}
    te = tags[tags.split == 'test']
    bal = te.groupby(['lat2', 'view2']).size().unstack()
    assert (bal.CC == bal.MLO).all(), 'test CC/MLO not balanced per laterality'
    folds = pd.read_csv(ROOT / 'outputs/19/fold_manifest.csv')
    assert len(folds) == 212 and folds.groupby('group_union').fold_group_union.nunique().max() == 1

    nb = json.loads((ROOT / 'notebooks/v7_dinov3_final_5090.ipynb').read_text())
    code_cells = [c['source'] for c in nb['cells'] if c['cell_type'] == 'code']
    assert all(not c['outputs'] and c['execution_count'] is None for c in nb['cells'] if c['cell_type'] == 'code')
    src = '\n\n'.join(code_cells)
    clean = '\n'.join(l for l in src.splitlines() if not l.startswith(('!', '%')))
    tree = ast.parse(clean, feature_version=(3, 11))
    assert not re.search(r'hf_[A-Za-z0-9]{20,}', src), 'hardcoded token'
    assert 'image.mode' not in src and ".mode ==" not in src, 'file-format prior must not appear'
    imports = [i for i, c in enumerate(code_cells) if re.search(r'^\s*(import|from) \S+', c, re.M)]
    assert len(imports) == 1, f'imports outside the Libraries cell: {imports}'
    nb_funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for path, names in [('eda/prep.py', ['breast_mask']), ('eda/prep_v6.py', ['read_gray', 'letterbox', 'prepare_array', 'preprocess']),
                        ('eda/prep_v7.py', None)]:
        for node in ast.parse((ROOT / path).read_text()).body:
            if isinstance(node, ast.FunctionDef) and (names is None or node.name in names):
                if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                    node.body = node.body[1:]
                assert ast.dump(node) == ast.dump(nb_funcs[node.name]), f'preprocessing drift: {node.name}'
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    defined = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)} | set(nb_funcs) \
        | {n.name for n in tree.body if isinstance(n, ast.ClassDef)} \
        | {a.asname or a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names} \
        | {a.arg for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.Lambda)) for a in n.args.args} \
        | {n.name for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.name}
    missing = names - defined - set(dir(__builtins__)) - {'display'}
    assert not missing, f'undefined names in notebook: {missing}'
    print('PASS: tissue_crop, robust_scale, augment/jitter/FM view, OCR tags + test balance, folds, notebook syntax, single import cell, no mode prior, source parity, names.')


if __name__ == '__main__':
    main()
