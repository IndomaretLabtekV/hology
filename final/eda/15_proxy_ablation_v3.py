"""15 - Does augmentation v3 (from 11-14) actually help under the purged spatial CV? CPU proxy as in 08
(UNet-ResNet18, half-res 512, crop 256, 6 epochs, folds 0 and 2) so numbers are comparable with 08_ablation.csv.
Configs: bank (= v2 recipe: contrast only) | v3 (contrast + seam .25 + haze_block .25 + fixed clouds .1, no blur, no
global haze) | v3w (v3 + coverage-balanced tile sampling from 13).
Val variants: clean, + regional haze (test-like mosaic block), + clouds. For each: IoU at 0.5 and the threshold that
maximises IoU (does a test-like shift push the best threshold down the way the LB does, and does v3 undo it?).
usage: python 15_proxy_ablation_v3.py <cfg> [folds] [epochs] [threads]"""
import sys, time, json
import numpy as np, pandas as pd, cv2, torch, torch.nn.functional as F
import segmentation_models_pytorch as smp
from common import *
from augs import restretch, fractal_noise, d4

CFG = sys.argv[1]; FOLDS = [int(f) for f in (sys.argv[2] if len(sys.argv) > 2 else "0,2").split(",")]
EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 16); torch.manual_seed(0)
S, CROP, BS, STEPS = 512, 256, 8, 40
d = np.load(CACHE / "half512.npz"); Xh, Yh, Nh = d["X"], d["Y"], d["N"].astype(bool)
folds = pd.read_csv(out("03d_folds.csv")); W = pd.read_csv(out("13_sampling_weights.csv")).w_balanced.values


def nod(img):
    return nodata_mask(img, 1024)  # half-res -> quarter area threshold


def contrast(img, rng, region=None):
    v = ~nod(img) if region is None else ~nod(img) & region
    if v.sum() < 100: return img
    o = img.astype(np.float32); mu = o[v].mean(0)
    g = np.exp(np.clip(rng.normal(0, .28), -.5, .5) + rng.normal(0, .04, 3)); o[v] = np.clip((o[v] - mu) * g + mu + rng.normal(0, 5, 3), 0, 255)
    return o.round().astype(np.uint8)


def block(h, w, rng):
    r = np.zeros((h, w), bool); k1, k2 = rng.integers(h // 5, 4 * h // 5, 2); m = rng.integers(3)
    if m == 0: r[:k1] = True
    elif m == 1: r[:, :k1] = True
    else: r[:k1, :k2] = True
    return ~r if rng.random() < .5 else r


def seam(img, rng):
    return restretch(contrast(img, rng, block(*img.shape[:2], rng)))


def haze_block(img, rng, strength=(0.05, 0.4)):
    reg = block(*img.shape[:2], rng) & ~nod(img); s = rng.uniform(*strength)
    tint = np.array([rng.uniform(140, 180), rng.uniform(160, 195), rng.uniform(205, 245)], np.float32)
    o = img.astype(np.float32); o[reg] = o[reg] * (1 - s) + tint * s
    return restretch(o.round().astype(np.uint8))


def cloud(img, rng, cover=(0.05, 0.3), opacity=(0.3, 0.8)):
    h, w = img.shape[:2]; n = fractal_noise(h, w, rng); t = np.quantile(n, 1 - rng.uniform(*cover))
    a = cv2.GaussianBlur(np.clip((n - t) / (1 - t + 1e-6) * 2.5, 0, 1) * rng.uniform(*opacity), (0, 0), 3)[..., None]
    o = img.astype(np.float32) * (1 - a) + np.array(rng.uniform(215, 250, 3), np.float32) * a; o[nod(img)] = 0
    return restretch(o.round().astype(np.uint8))


AUG = {"bank": [(.5, contrast)],
       "v3": [(.5, contrast), (.25, seam), (.25, haze_block), (.1, cloud)]}
AUG["v3w"] = AUG["v3"]
MEAN, STD = np.array([0.485, 0.456, 0.406]) * 255, np.array([0.229, 0.224, 0.225]) * 255
norm = lambda x: torch.from_numpy(((x.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy()).float()


def sample(ids, p, rng):
    i = rng.choice(ids, p=p); x, y = Xh[i], Yh[i]
    for _ in range(10):
        y0, x0 = rng.integers(0, S - CROP + 1, 2)
        if (x[y0:y0 + CROP, x0:x0 + CROP].max(-1) > 0).mean() > .2: break
    x, y = d4(x[y0:y0 + CROP, x0:x0 + CROP].copy(), y[y0:y0 + CROP, x0:x0 + CROP].copy(), rng)
    for pr, f in AUG[CFG]:
        if rng.random() < pr: x = f(x, rng)
    return norm(x), torch.from_numpy(y.astype(np.float32))


def predict(model, imgs):
    model.eval(); o = []
    with torch.no_grad():
        for x in imgs:
            t = norm(x)[None]; pr = 0
            for dims in ([], [2], [3]):
                r = torch.sigmoid(model(torch.flip(t, dims) if dims else t)); pr = pr + (torch.flip(r, dims) if dims else r)
            o.append((pr / 3)[0, 0].numpy())
    return np.stack(o)


def curve(prob, y, nd):
    g = y.astype(bool); res = {}
    for t in (.1, .2, .3, .4, .5, .6):
        pr = (prob > t) & ~nd; res[t] = (pr & g).sum() / ((pr | g).sum() + 1e-9)
    return res


rows = []
for k in FOLDS:
    t0 = time.time(); rng = np.random.default_rng(k)
    tr = folds.index[(folds.fold != k) & ~folds[f"purge_{k}"]].values; va = folds.index[folds.fold == k].values
    p = W[tr] / W[tr].sum() if CFG == "v3w" else np.full(len(tr), 1 / len(tr))
    model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EPOCHS * STEPS, pct_start=.1)
    dice = smp.losses.DiceLoss("binary")
    for ep in range(EPOCHS):
        model.train()
        for _ in range(STEPS):
            b = [sample(tr, p, rng) for _ in range(BS)]
            x = torch.stack([a for a, _ in b]); y = torch.stack([c for _, c in b])[:, None]
            lo = model(x); loss = F.binary_cross_entropy_with_logits(lo, y) + dice(lo, y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    vx, vy, vn = Xh[va], Yh[va], Nh[va]; rs = np.random.default_rng(1000 + k)
    variants = {"clean": vx, "haze": np.stack([haze_block(x, rs, (0.2, 0.45)) for x in vx]),
                "clouds": np.stack([cloud(x, rs, (0.15, 0.35), (0.5, 0.85)) for x in vx])}
    r = dict(cfg=CFG, fold=k, secs=round(time.time() - t0))
    for n, v in variants.items():
        c = curve(predict(model, v), vy, vn); best = max(c, key=c.get)
        r[f"{n}_t05"], r[f"{n}_t01"], r[f"{n}_best"], r[f"{n}_bestthr"] = c[.5], c[.1], c[best], best
    rows.append(r); print(json.dumps({a: (round(b, 4) if isinstance(b, float) else b) for a, b in r.items()}), flush=True)
df = pd.DataFrame(rows); f = out("15_ablation_v3.csv"); df.to_csv(f, mode="a", header=not f.exists(), index=False)
print(CFG, "MEAN:", df.drop(columns=["cfg", "fold"]).mean().round(4).to_dict(), flush=True)
