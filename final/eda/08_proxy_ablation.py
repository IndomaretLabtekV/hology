"""08 - Proxy-model ablation on CPU: does each preprocessing / augmentation choice actually help under the
spatial purged CV? Small UNet-ResNet18 (ImageNet) at half resolution (512px tiles, 256px crops).
Reports micro-IoU on clean val tiles and on stressed val tiles (real-gain affine, synthetic clouds).
usage: python 08_proxy_ablation.py <config> [folds e.g. 0,1,2,3,4] [epochs]"""
import sys, time, json
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn, torch.nn.functional as F
import segmentation_models_pytorch as smp
from common import *
from augs import *

CFG = sys.argv[1]; FOLDS = [int(f) for f in (sys.argv[2] if len(sys.argv) > 2 else "0,1,2,3,4").split(",")]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 12
torch.set_num_threads(16); torch.manual_seed(0)
S, CROP, BS, STEPS = 512, 256, 8, 40

p = CACHE / "half512.npz"
if not p.exists():
    Xh = np.stack([cv2.resize(np.ascontiguousarray(x), (S, S), interpolation=cv2.INTER_AREA) for x in train_images()])
    Yh = np.stack([cv2.resize(y, (S, S), interpolation=cv2.INTER_NEAREST) for y in train_masks()])
    Nh = np.stack([cv2.resize(nodata_mask(x).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) for x in train_images()])
    np.savez(p, X=Xh, Y=Yh, N=Nh)
d = np.load(p); Xh, Yh, Nh = d["X"], d["Y"], d["N"].astype(bool)
folds = pd.read_csv(out("03d_folds.csv"))
bank = load_gain_bank()

AUG = {  # name -> list of (prob, fn(img, rng) -> img)
    "d4": [],
    "bank": [(.5, lambda x, r: bank_affine(x, bank, r))],
    "hsv": [(.5, lambda x, r: hsv_jitter(x, r))],
    "rot": [],
    "full_v1": [(.5, lambda x, r: bank_affine(x, bank, r)), (.2, lambda x, r: seam_affine(x, bank, r)),
                (.15, lambda x, r: haze(x, r)), (.15, lambda x, r: clouds(x, r)), (.2, lambda x, r: blur(x, r))],
    "full_v2": [(.5, lambda x, r: bank_affine(x, bank, r)), (.2, lambda x, r: restretch(seam_affine(x, bank, r))),
                (.15, lambda x, r: restretch(haze(x, r))), (.15, lambda x, r: restretch(clouds(x, r, (0.05, 0.3), (0.3, 0.8)))),
                (.2, lambda x, r: restretch(blur(x, r, (0.3, 0.8))))],
    "full_v2_rot": None, "full_v2_znorm": None,
}
AUG["full_v2_rot"] = AUG["full_v2"]; AUG["full_v2_znorm"] = AUG["full_v2"]
GEOM_ROT = CFG in ("rot", "full_v2_rot")
ZNORM = CFG.endswith("znorm")
MEAN, STD = np.array([0.485, 0.456, 0.406]) * 255, np.array([0.229, 0.224, 0.225]) * 255

def norm(x):
    x = x.astype(np.float32)
    if ZNORM:
        v = x.max(-1) > 0
        if v.sum() > 100: x = (x - x[v].mean(0)) / (x[v].std(0) + 1e-3) * STD + MEAN; x[~v] = 0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy()).float()

def sample(ids, rng):
    i = ids[rng.integers(len(ids))]; x, y = Xh[i], Yh[i]
    for _ in range(10):
        y0, x0 = rng.integers(0, S - CROP + 1, 2)
        if (x[y0:y0 + CROP, x0:x0 + CROP].max(-1) > 0).mean() > .2: break
    x, y = x[y0:y0 + CROP, x0:x0 + CROP].copy(), y[y0:y0 + CROP, x0:x0 + CROP].copy()
    x, y = d4(x, y, rng)
    if GEOM_ROT and rng.random() < .5: x, y = rotate_scale(x, y, rng, scale=(0.85, 1.2))
    for pr, f in AUG[CFG]:
        if rng.random() < pr: x = f(x, rng)
    return norm(x), torch.from_numpy(y.astype(np.float32))

def predict(model, imgs):
    model.eval(); out_ = []
    with torch.no_grad():
        for x in imgs:
            t = norm(x)[None]
            pr = 0
            for dims in ([], [2], [3]):
                o = torch.sigmoid(model(torch.flip(t, dims) if dims else t))
                pr = pr + (torch.flip(o, dims) if dims else o)
            out_.append((pr / 3)[0, 0].numpy())
    return np.stack(out_)

def miou(prob, y, nod, thr=.5):
    pr = (prob > thr) & ~nod; g = y.astype(bool)
    return (pr & g).sum() / ((pr | g).sum() + 1e-9)

rows = []; rng_stress = np.random.default_rng(123)
for k in FOLDS:
    t0 = time.time(); rng = np.random.default_rng(k)
    tr_ids = folds.index[(folds.fold != k) & ~folds[f"purge_{k}"]].values; va_ids = folds.index[folds.fold == k].values
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EPOCHS * STEPS, pct_start=.1)
    dice = smp.losses.DiceLoss("binary")
    for ep in range(EPOCHS):
        model.train()
        for _ in range(STEPS):
            b = [sample(tr_ids, rng) for _ in range(BS)]
            x = torch.stack([a for a, _ in b]); y = torch.stack([c for _, c in b])[:, None]
            lo = model(x); loss = F.binary_cross_entropy_with_logits(lo, y) + dice(lo, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    vx, vy, vn = Xh[va_ids], Yh[va_ids], Nh[va_ids]
    rs = np.random.default_rng(1000 + k)
    vx_aff = np.stack([bank_affine(x, bank, rs) for x in vx])
    vx_cld = np.stack([restretch(clouds(x, rs, (0.15, 0.35), (0.5, 0.85))) for x in vx])
    p_clean, p_aff, p_cld = predict(model, vx), predict(model, vx_aff), predict(model, vx_cld)
    r = dict(cfg=CFG, fold=k, iou=miou(p_clean, vy, vn), iou_affine=miou(p_aff, vy, vn), iou_clouds=miou(p_cld, vy, vn),
             iou_no_nodata_rule=miou(p_clean, vy, np.zeros_like(vn)), secs=round(time.time() - t0))
    for thr in (.3, .4, .6): r[f"iou_t{thr}"] = miou(p_clean, vy, vn, thr)
    rows.append(r); print(json.dumps({k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in r.items()}), flush=True)
    np.save(CACHE / f"oof_{CFG}_f{k}.npy", p_clean.astype(np.float16))
df = pd.DataFrame(rows)
f = out("08_ablation.csv"); df.to_csv(f, mode="a", header=not f.exists(), index=False)
print(CFG, "MEAN:", df.drop(columns=["cfg", "fold"]).mean().round(4).to_dict())
