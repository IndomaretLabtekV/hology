"""Stress mask thresholds; a looser mask is an uncertainty reference, not ground-truth anatomy."""
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from deep_common import grid, manifest, output, plt
from prep import breast_mask
from prep_v6 import read_gray


def masks(path):
    a=read_gray(path,4,True)
    b=cv2.GaussianBlur(a,(5,5),0)
    otsu=cv2.threshold(b,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]
    bg=int(np.argmax(np.bincount(b.ravel(),minlength=256)[:max(1,int(otsu))]))
    loose=breast_mask(a,thr=bg+max(8,.35*(otsu-bg)))
    base=breast_mask(a)
    strict=breast_mask(a,thr=bg+max(22,.65*(otsu-bg)))
    return a,loose,base,strict


def measure(row):
    a,loose,base,strict=masks(row.path)
    k=max(3,round(15*a.shape[0]/1185))
    distance=cv2.distanceTransform((~base).astype(np.uint8),cv2.DIST_L2,3)
    alpha=np.clip(1-distance/k,0,1);alpha=alpha*alpha*(3-2*alpha)
    guard=np.clip(2-distance/k,0,1);guard=guard*guard*(3-2*guard)
    uncertain=loose&~base
    return dict(image_id=row.image_id,label=row.label,split=row.split,base_fraction=base.mean(),
                loose_extra_fraction=uncertain.sum()/max(loose.sum(),1),
                strict_removed_fraction=(base&~strict).sum()/max(base.sum(),1),
                loose_signal_attenuated=float(((1-alpha[loose])*a[loose]).sum()/max(a[loose].sum(),1)),
                guard_loose_signal_attenuated=float(((1-guard[loose])*a[loose]).sum()/max(a[loose].sum(),1)))


def main():
    out=output(24);data=manifest()
    with ThreadPoolExecutor(4) as pool:result=pd.DataFrame(pool.map(measure,data.itertuples(index=False)))
    result.to_csv(out/'mask_sensitivity.csv',index=False)
    print(result.drop(columns=['image_id','label','split']).describe().round(4).to_string())
    worst=result.sort_values('loose_signal_attenuated',ascending=False).head(12)
    print('Most sensitive masks (not confirmed failures):\n',worst.round(4).to_string(index=False))
    paths=data.set_index('image_id').path
    images=[];titles=[]
    for row in worst.itertuples():
        a,loose,base,strict=masks(paths[row.image_id])
        for name,m in [('raw',None),('loose',loose),('base',base),('strict',strict)]:
            image=a if m is None else np.where(m,a,0)
            images.append(image);titles.append(f'{row.image_id} {name}\nloose loss={row.loose_signal_attenuated:.3f}')
    grid(images,titles,out/'sensitive_masks.png',cols=4)
    fig,ax=plt.subplots(figsize=(7,4))
    for name in ['loose_signal_attenuated','guard_loose_signal_attenuated']:
        ax.hist(result[name],bins=25,alpha=.5,label=name)
    ax.legend();ax.set_xlabel('Fraction of looser-mask intensity attenuated (proxy)')
    fig.tight_layout();fig.savefig(out/'mask_sensitivity.png',dpi=120);plt.close(fig)


if __name__=='__main__':
    cv2.setNumThreads(1)
    main()
