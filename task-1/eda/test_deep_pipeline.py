"""Run with .venv/bin/python eda/test_deep_pipeline.py. No notebook cell is executed."""
import ast
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from prep_v6 import prepare_array, rotate, rotation_matrix, gamma_intensity, augment

ROOT=Path(__file__).resolve().parents[1]


def main():
    image=np.zeros((512,384),np.uint8)
    cv2.ellipse(image,(0,256),(230,210),0,0,360,150,-1)
    prepared,mask,support,_=prepare_array(image,(512,384))
    assert prepared.dtype==np.uint8 and prepared.shape==image.shape
    assert np.all(prepared[mask]==image[mask]), 'Soft mask changed tissue intensities.'
    assert not prepared[~support].any(), 'Exterior texture survived.'
    for angle in [-5,0,5]:
        m=rotation_matrix(image.shape,angle,fit=True)
        corners=m@np.array([[0,383,0,383],[0,0,511,511],[1,1,1,1]])
        assert corners.min()>=-1e-5 and corners[0].max()<=383+1e-5 and corners[1].max()<=511+1e-5
    assert np.array_equal(rotate(image,0,'constant',True),image)
    for gamma in [.9,1,1.1]:
        ramp=gamma_intensity(np.arange(256,dtype=np.uint8)[None],gamma)
        assert ramp[0,0]==0 and ramp[0,-1]==255 and np.all(np.diff(ramp.astype(int))>=0)
    assert np.array_equal(augment(image,np.random.default_rng(42)),augment(image,np.random.default_rng(42)))
    try:
        prepare_array(np.zeros((2,2),np.uint8))
    except ValueError:
        pass
    else:
        raise AssertionError('Tiny invalid input accepted.')
    manifest=pd.read_csv(ROOT/'outputs/19/fold_manifest.csv')
    assert manifest.image_id.is_unique and len(manifest)==212
    assert manifest.groupby('group_union').fold_group_union.nunique().max()==1
    nb_path=ROOT/'notebooks/v6_dinov3_eda_5090.ipynb'
    assert nb_path.exists(), 'GPU notebook has not been generated.'
    nb=json.loads(nb_path.read_text())
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in nb['cells'] if c['cell_type']=='code')
    code='\n\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    clean='\n'.join(line for line in code.splitlines() if not line.startswith(('!','%')))
    ast.parse(clean, feature_version=(3,11))
    assert not re.search(r'hf_[A-Za-z0-9]{20,}', code), 'Hardcoded credential in notebook.'
    assert "assert torch.cuda.is_available()" in code
    # The builder must embed the exact tested preprocessing function bodies, not a hand-maintained fork.
    parsed=ast.parse(clean)
    for name in ['prepare_array','preprocess','augment','rotation_matrix']:
        a=next(x for x in ast.parse((ROOT/'eda/prep_v6.py').read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name)
        b=next(x for x in parsed.body if isinstance(x,ast.FunctionDef) and x.name==name)
        assert ast.dump(a)==ast.dump(b), f'Notebook preprocessing drift: {name}'
    print('PASS: preprocessing, rotations, intensity, groups, notebook syntax and source parity. No notebook execution.')


if __name__=='__main__':
    main()
