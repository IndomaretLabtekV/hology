"""10 - Is the val->LB gap explained by test composition (more settlement, less sawah per tile)?
Decodes a submission, compares predicted sawah fraction per test tile with the train distribution, then
re-weights OOF tiles (proxy OOF) to the test fraction distribution and reports the expected micro-IoU.
usage: python 10_test_composition.py <submission.csv> [oof_cfg=bank]"""
import sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

sub = pd.read_csv(sys.argv[1], keep_default_na=False); cfg = sys.argv[2] if len(sys.argv) > 2 else "bank"
def frac_of(rle):
    s = list(map(int, rle.split())); return sum(s[1::2]) / 1024 ** 2
T = test_images()
pf = np.array([frac_of(r) for r in sub.EncodedPixels])
valid_te = np.array([1 - nodata_mask(t).mean() for t in T])
Y = train_masks(); X = train_images()
tf = Y.reshape(len(Y), -1).mean(1); valid_tr = np.array([1 - nodata_mask(x).mean() for x in X])
print("sawah fraction per tile   train GT: mean %.3f median %.3f | test PRED: mean %.3f median %.3f" % (tf.mean(), np.median(tf), pf.mean(), np.median(pf)))
print("sawah share of VALID area train GT %.3f | test PRED %.3f" % (Y.sum() / (valid_tr.sum() * 1024 ** 2), pf.sum() / valid_te.sum()))
bins = [-.01, .05, .15, .35, .6, 1.01]
print("tile share by fraction bin:\n", pd.DataFrame({"train_GT": pd.Series(pd.cut(tf, bins)).value_counts(normalize=True, sort=False),
                                                     "test_PRED": pd.Series(pd.cut(pf, bins)).value_counts(normalize=True, sort=False)}).round(3).to_string())

# expected micro-IoU if OOF tiles were drawn with the test fraction mix (importance weights by bin)
d = np.load(CACHE / "half512.npz"); Yh, Nh = d["Y"].astype(bool), d["N"].astype(bool)
folds = pd.read_csv(out("03d_folds.csv")); rows = []
for k in range(5):
    f = CACHE / f"oof_{cfg}_f{k}.npy"
    if not f.exists(): continue
    for i, p in zip(folds.index[folds.fold == k], np.load(f)):
        pr = (p > .5) & ~Nh[i]; g = Yh[i]
        rows.append(dict(frac=g.mean(), tp=(pr & g).sum(), fp=(pr & ~g).sum(), fn=(~pr & g).sum()))
O = pd.DataFrame(rows); O["bin"] = pd.cut(O.frac, bins)
w_test = pd.Series(pd.cut(pf, bins)).value_counts(normalize=True, sort=False); w_oof = O.bin.value_counts(normalize=True, sort=False)
O["w"] = O.bin.map((w_test / w_oof).fillna(0)).astype(float)
plain = O.tp.sum() / (O.tp + O.fp + O.fn).sum()
rew = (O.w * O.tp).sum() / (O.w * (O.tp + O.fp + O.fn)).sum()
print(f"proxy OOF micro-IoU: as-is {plain:.4f} | re-weighted to test fraction mix {rew:.4f} (drop {plain - rew:+.4f})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(tf, bins=25, alpha=.6, density=True, label="train GT"); ax[0].hist(pf, bins=25, alpha=.6, density=True, label="test pred")
ax[0].set_title("sawah fraction per tile"); ax[0].legend()
ax[1].scatter(valid_te, pf, s=8); ax[1].set_xlabel("valid fraction"); ax[1].set_ylabel("pred sawah frac"); ax[1].set_title("test tiles")
plt.tight_layout(); plt.savefig(out("10_test_composition.png"), dpi=100)
