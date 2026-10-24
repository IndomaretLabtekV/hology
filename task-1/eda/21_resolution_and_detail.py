"""Sampling limits: real-image round trips and controlled synthetic dots, not lesion annotations."""
import cv2
import numpy as np
import pandas as pd

from deep_common import grid, manifest, output, plt, samples
from prep_v6 import letterbox, prepare_array, read_gray


def main():
    out = output(21)
    data = manifest(); picks = samples(data)
    rows = []
    sizes = [(640, 480), (1024, 768), (1536, 1152)]
    for row in picks.itertuples():
        raw = read_gray(row.path, 2, harmonize=True)
        ref, mask, _, _ = prepare_array(raw, out=raw.shape, variant='soft')
        ims, titles = [], []
        for size in sizes:
            small = letterbox(ref, size)
            # Original aspect is approximately 4:3; undo letterboxing using exact fitted dimensions.
            ratio = min(size[0]/ref.shape[0], size[1]/ref.shape[1])
            h, w = round(ref.shape[0]*ratio), round(ref.shape[1]*ratio)
            top = (size[0]-h)//2
            back = cv2.resize(small[top:top+h, :w], ref.shape[::-1], interpolation=cv2.INTER_LINEAR)
            delta = (back.astype(float)-ref)[mask]
            grad_ref = cv2.Sobel(ref.astype(np.float32), cv2.CV_32F, 1, 0)
            grad_back = cv2.Sobel(back.astype(np.float32), cv2.CV_32F, 1, 0)
            rows.append(dict(image_id=row.image_id, label=row.label, height=size[0],
                             mse_tissue=np.mean(delta**2), gradient_retention=np.std(grad_back[mask])/max(np.std(grad_ref[mask]), 1e-6),
                             patch_tokens=(size[0]//16)*(size[1]//16), approximate_native_pixels_per_token=16*2/ratio))
            yy, xx = np.where(mask)
            cy, cx = int(np.median(yy)), int(np.median(xx))
            crop = back[max(0,cy-160):cy+160, max(0,cx-160):cx+160]
            ims.append(crop); titles.append(f'{row.label} {row.image_id}\n{size[0]}x{size[1]} round trip')
        grid(ims, titles, out / f'detail_{row.image_id[:-4]}.png', cols=3)
    res = pd.DataFrame(rows);res.to_csv(out/'real_detail.csv',index=False)
    print(res.groupby('height').mean(numeric_only=True).round(3).to_string())
    # Fixed signal plus fixed constant background, so noise cannot masquerade as retained detail.
    h,w = 2370,1770
    yy,xx=np.mgrid[:h,:w];base=np.full((h,w),80,dtype=np.float32)
    synthetic=[]; dots=[]
    for j,sigma in enumerate([1,2,4,8]):
        cy,cx=500+j*400,800
        signal=base+100*np.exp(-((yy-cy)**2+(xx-cx)**2)/(2*sigma**2))
        for size in sizes:
            reduced=cv2.resize(signal,size[::-1],interpolation=cv2.INTER_AREA)
            restored=cv2.resize(reduced,(w,h),interpolation=cv2.INTER_LINEAR)
            peak=float(restored[cy-12:cy+13,cx-12:cx+13].max()-80)
            dots.append(dict(sigma_at_half_resolution=sigma,height=size[0],retained_peak_fraction=peak/100))
            synthetic.append(restored[cy-24:cy+25,cx-24:cx+25])
    dot=pd.DataFrame(dots);dot.to_csv(out/'synthetic_dots.csv',index=False)
    print('Synthetic peak retention (not clinical lesions):\n',dot.pivot(index='sigma_at_half_resolution',columns='height',values='retained_peak_fraction').round(3).to_string())
    grid(synthetic,[f'sigma={r.sigma_at_half_resolution}, H={r.height}\npeak={r.retained_peak_fraction:.2f}' for r in dot.itertuples()],out/'synthetic_dot_retention.png',cols=3)
    fig,ax=plt.subplots(figsize=(6,4))
    for sigma,d in dot.groupby('sigma_at_half_resolution'):
        ax.plot(d.height,d.retained_peak_fraction,'o-',label=f'sigma {sigma} pixels')
    ax.set(xlabel='Input height',ylabel='Peak contrast retained',ylim=(0,1));ax.legend();fig.tight_layout();fig.savefig(out/'resolution_tradeoff.png',dpi=120);plt.close(fig)


if __name__=='__main__':
    cv2.setNumThreads(1)
    main()
