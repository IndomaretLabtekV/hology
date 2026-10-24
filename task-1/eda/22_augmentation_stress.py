"""Controlled matched-draw rotation and photometric stress tests on train images only."""
import json

import cv2
import numpy as np
import pandas as pd

from deep_common import ROOT, grid, manifest, output, plt, samples
from prep_v6 import augment, gamma_intensity, rotation_matrix, rotate


def main():
    out = output(22)
    data=manifest();train=data[data.split=='train'];rng=np.random.default_rng(42)
    rows=[]
    rotations=[('constant15',15,'constant',False),('reflect15',15,'reflect',False),
               ('reflect5',5,'reflect',False),('fit_reflect5',5,'reflect',True),('fit_constant5',5,'constant',True)]
    for row in train.itertuples():
        image=cv2.imread(str(ROOT/'outputs/20/cache_soft'/row.image_id.replace('.jpg','.png')),0)
        if image is None: raise FileNotFoundError(row.image_id)
        image=cv2.resize(image,(384,512),interpolation=cv2.INTER_AREA)
        mask=(image>10).astype(np.uint8);y,x=np.where(mask)
        coords=np.stack([x,y,np.ones(len(x))])
        draws=rng.uniform(-1,1,3)
        for name,degrees,border,fit in rotations:
            for draw in draws:
                angle=float(draw*degrees);mat=rotation_matrix(image.shape,angle,fit)
                mapped=mat@coords
                retained=((mapped[0]>=0)&(mapped[0]<=383)&(mapped[1]>=0)&(mapped[1]<=511)).mean()
                real=rotate(mask,angle,'constant',fit,nearest=True)>0
                mirrored=rotate(mask,angle,'reflect',fit,nearest=True)>0
                extra=(mirrored & ~real).sum()/max(real.sum(),1) if border=='reflect' else 0
                valid=rotate(np.ones_like(mask),angle,'constant',fit,nearest=True)>0
                rows.append(dict(image_id=row.image_id,label=row.label,kind='rotation',variant=name,
                                 tissue_lost=1-retained,reflected_extra=extra,invalid_canvas=(~valid).mean(),
                                 scale=float(np.sqrt(np.linalg.det(mat[:,:2])))))
        f=image.astype(float)/255
        for name in ['jitter_03','gamma_wide','gamma_mild']:
            for _ in range(3):
                if name=='jitter_03':
                    z=np.clip(f*rng.uniform(.7,1.3),0,1);contrast=rng.uniform(.7,1.3)
                    z=np.clip(contrast*z+(1-contrast)*z.mean(),0,1)
                else:
                    limits=(.7,1.4) if name=='gamma_wide' else (.9,1.1)
                    z=gamma_intensity(image,float(rng.uniform(*limits))).astype(float)/255
                rows.append(dict(image_id=row.image_id,label=row.label,kind='intensity',variant=name,
                                 background_lift=float((z[image==0]>.01).mean()),
                                 newly_clipped=float(((z>=.999)&(f<.99)&(mask>0)).sum()/max(mask.sum(),1)),
                                 tissue_mae=float(np.abs(z-f)[mask>0].mean())))
    res=pd.DataFrame(rows);res.to_csv(out/'augmentation_metrics.csv',index=False)
    cols=['tissue_lost','reflected_extra','invalid_canvas','scale','background_lift','newly_clipped','tissue_mae']
    summary=res.groupby('variant')[cols].agg(['mean','max']);summary.to_csv(out/'summary.csv')
    print(summary.round(5).to_string())
    selected=samples(data)
    for row in selected.itertuples():
        image=cv2.imread(str(ROOT/'outputs/20/cache_soft'/row.image_id.replace('.jpg','.png')),0)
        ims=[image]+[rotate(image,degrees,border,fit) for _,degrees,border,fit in rotations]
        grid(ims,['original']+[x[0] for x in rotations],out/f'rotation_{row.label}_{row.image_id[:-4]}.png')
        f=image.astype(float)/255;naive=np.clip(.7*f+.3*f.mean(),0,1)
        grid([image,(naive*255).astype(np.uint8),gamma_intensity(image,.7),gamma_intensity(image,.9),gamma_intensity(image,1.1),augment(image,np.random.default_rng(42))],
             ['original','contrast 0.7, background lifted','gamma 0.7','gamma 0.9','gamma 1.1','selected augmentation'],out/f'intensity_{row.label}_{row.image_id[:-4]}.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,col in zip(axes,['tissue_lost','reflected_extra','background_lift']):
        res.groupby('variant')[col].mean().dropna().plot.bar(ax=ax);ax.set_title(col)
    fig.tight_layout();fig.savefig(out/'augmentation_comparison.png',dpi=120);plt.close(fig)
    (out/'decision.json').write_text(json.dumps(dict(degrees=5,gamma=[.9,1.1],fit=True,border='constant',
        warning='Metrics measure geometry/intensity only. No claim of label preservation or improved F1 until GPU ablation.'),indent=2))


if __name__=='__main__':
    cv2.setNumThreads(1)
    main()
