"""EDA 17: re-score every saved feature set with the CORRECT patient grouping (Mammo-FM+OCR pairs, precision ~0.93).
EDA 16 showed the old grouping (precision ~0.75) inflated OOF F1 by ~0.11 via cross-view leakage.
Reports single-view and pair-averaged macro F1, plus AUCs. Benign-vs-Malignant AUC is the leak-free metric
(both classes share file format/export batch); Normal AUC can still carry the JPEG/export signature."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT

tr = pd.read_csv(ROOT / "data/train.csv").merge(pd.read_csv(ROOT / "outputs/16/train_pairs_fm.csv"), on="image_id")
C = ["Normal", "Benign", "Malignant"]; y = tr.label.map(C.index).values; grp = tr.pair_fm.values
order = pd.read_csv(ROOT / "outputs/05/groups.csv"); order = order[order.label != "?"].image_id.tolist()
rows = []
for f in sorted((ROOT / "outputs/12").glob("feat_*.npy")):
    X = pd.DataFrame(np.load(f), index=order).loc[tr.image_id].values
    f1s, f1p, P = [], [], np.zeros((len(y), 3))
    for s in range(5):
        p = np.zeros((len(y), 3))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, grp):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
            p[b] = clf.fit(X[a], y[a]).predict_proba(X[b])
        pp = pd.DataFrame(p).groupby(grp).transform("mean").values
        f1s.append(f1_score(y, p.argmax(1), average="macro")); f1p.append(f1_score(y, pp.argmax(1), average="macro")); P += p / 5
    bm = y > 0
    rows.append(dict(features=f.stem.replace("feat_", ""), f1=round(np.mean(f1s), 3), f1_pairavg=round(np.mean(f1p), 3), sd=round(np.std(f1p), 3),
                     auc_normal=round(roc_auc_score(y == 0, P[:, 0]), 3), auc_BM=round(roc_auc_score(y[bm] == 2, P[bm, 2] / (P[bm, 1] + P[bm, 2])), 3)))
res = pd.DataFrame(rows).sort_values("auc_BM", ascending=False)
res.to_csv(ROOT / "outputs/12/reprobe_pairs.csv", index=False)
print(res.to_string(index=False))
