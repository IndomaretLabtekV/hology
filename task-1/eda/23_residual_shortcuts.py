"""Residual scalar associations and paired JPEG interventions. No model fitting or test labels."""
from concurrent.futures import ThreadPoolExecutor
import json

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deep_common import ROOT, CLASSES, grid, manifest, output, plt
from prep_v6 import prepare_array, read_gray


def scalar_features(image, mask, support):
    residual=image.astype(np.float32)-cv2.GaussianBlur(image.astype(np.float32),(0,0),1)
    bg=image[~support]
    return dict(tissue_median=float(np.median(image[mask])), tissue_noise=float(residual[mask].std()),
                bg_std=float(bg.std()) if len(bg) else 0, area=float(mask.mean()))


def inspect(row):
    ordinary=read_gray(row.path,2,False)
    harmonized=read_gray(row.path,2,True)
    rows=[]
    for name, raw, variant in [('raw',ordinary,'raw'),('soft_no_harmonize',ordinary,'soft'),('soft_harmonized',harmonized,'soft')]:
        image,mask,support,_=prepare_array(raw)
        if variant=='raw':image,mask,support,_=prepare_array(raw,variant='raw')
        rows.append(dict(image_id=row.image_id,variant=name,**scalar_features(image,mask,support)))
    original,mask,_,_=prepare_array(ordinary)
    corrected,_,_,_=prepare_array(harmonized)
    delta=np.abs(corrected.astype(float)-original)
    return rows,dict(image_id=row.image_id,harmonize_mae_tissue=delta[mask].mean(),harmonize_max_delta=delta.max())


def main():
    out=output(23);data=manifest();train=data[data.split=='train']
    with ThreadPoolExecutor(4) as pool:results=list(pool.map(inspect,train.itertuples(index=False)))
    records=pd.DataFrame([r for rows,_ in results for r in rows]).merge(train[['image_id','label']],on='image_id')
    change=pd.DataFrame([r for _,r in results]).merge(train[['image_id','label']],on='image_id')
    records.to_csv(out/'scalar_features.csv',index=False);change.to_csv(out/'jpeg_intervention.csv',index=False)
    groups=pd.read_csv(ROOT/'outputs/19/fold_manifest.csv').set_index('image_id').group_union
    rows=[];rng=np.random.default_rng(42)
    for variant,d in records.groupby('variant'):
        for task in ['Normal','BM']:
            sub=d if task=='Normal' else d[d.label!='Normal']
            y=(sub.label==('Normal' if task=='Normal' else 'Malignant')).to_numpy()
            g=groups.loc[sub.image_id].to_numpy();unique=np.unique(g)
            for feature in ['tissue_median','tissue_noise','bg_std','area']:
                x=sub[feature].to_numpy();auc=roc_auc_score(y,x);bootstrap=[]
                for _ in range(300):
                    chosen=rng.choice(unique,len(unique),replace=True)
                    idx=np.concatenate([np.flatnonzero(g==k) for k in chosen])
                    if len(np.unique(y[idx]))==2:bootstrap.append(roc_auc_score(y[idx],x[idx]))
                rows.append(dict(variant=variant,task=task,feature=feature,auc=auc,
                                 lo=np.quantile(bootstrap,.025),hi=np.quantile(bootstrap,.975)))
    aucs=pd.DataFrame(rows);aucs.to_csv(out/'scalar_auc.csv',index=False)
    print(aucs.round(3).to_string(index=False));print('JPEG intervention:\n',change.groupby('label').mean(numeric_only=True).round(4).to_string())
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,task in zip(axes,['Normal','BM']):
        part=aucs[aucs.task==task].reset_index(drop=True)
        ax.errorbar(part.auc,np.arange(len(part)),xerr=[part.auc-part.lo,part.hi-part.auc],fmt='o')
        ax.set_yticks(np.arange(len(part)),part.variant+' / '+part.feature,fontsize=7)
        ax.axvline(.5,color='gray');ax.set_title(f'{task}: scalar AUC, exploratory group bootstrap');ax.set_xlim(0,1)
    fig.tight_layout();fig.savefig(out/'residual_associations.png',dpi=120);plt.close(fig)
    picks=[train[train.label==c].iloc[0] for c in CLASSES]
    images=[];titles=[]
    for row in picks:
        raw=read_gray(row.path,2,True);image,m,s,_=prepare_array(raw)
        original=prepare_array(raw,variant='raw')[0]
        high=image.astype(float)-cv2.GaussianBlur(image.astype(float),(0,0),2)
        for name,a in [('soft',image),('exterior',np.where(s,0,original)),('silhouette',m.astype(np.uint8)*255),('highpass display',np.where(m,np.clip(128+high*4,0,255),0))]:
            images.append(a);titles.append(f'{row.label}\n{name}')
    grid(images,titles,out/'negative_controls.png',cols=4)
    (out/'interpretation.json').write_text(json.dumps(dict(
        warning='Univariate associations are not held-out model performance. Anatomy can cause tissue associations; absence of scalar associations cannot prove no deep-model shortcut.',
        confidence_intervals='300 bootstrap resamples of inferred union groups; exploratory, no multiplicity correction.'),indent=2))


if __name__=='__main__':
    cv2.setNumThreads(1)
    main()
