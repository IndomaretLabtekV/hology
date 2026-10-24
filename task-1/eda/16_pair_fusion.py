"""EDA 16: patient-pair fusion. Both views of a case share one label, so averaging the two view predictions
should help IF pairs are right. Measures (a) pairing precision with Mammo-FM features vs ConvNeXt,
(b) OOF macro F1 of the frozen Mammo-FM probe with and without pair averaging,
(test pairing is not simulated: random train subsets break true pairs, which test never does)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
sys.path.insert(0, str(Path(__file__).parent))
from prep import ROOT

OUT = ROOT / "outputs/16"; OUT.mkdir(parents=True, exist_ok=True)
FEAT = "feat_mammo_fm_fm_notext_s2_h"
df = pd.read_csv(ROOT / "outputs/13/pairs.csv")
tr = df[df.split == "train"].reset_index(drop=True)
C = ["Normal", "Benign", "Malignant"]; y = tr.label.map(C.index).values
order = pd.read_csv(ROOT / "outputs/05/groups.csv"); order = order[order.label != "?"].image_id.tolist()
X = pd.DataFrame(np.load(ROOT / f"outputs/12/{FEAT}.npy"), index=order).loc[tr.image_id].values
E = pd.DataFrame(np.load(ROOT / "outputs/05/emb_convnext.npy"), index=pd.read_csv(ROOT / "outputs/05/groups.csv").image_id).loc[tr.image_id].values
z = lambda M: (M - M.mean()) / M.std()
cos = lambda A: (A / np.linalg.norm(A, axis=1, keepdims=True)) @ (A / np.linalg.norm(A, axis=1, keepdims=True)).T
Xs = StandardScaler().fit_transform(X)


def match(S, cells):
    pairs = []
    for idx in cells:
        cc = [i for i in idx if tr.view2[i] == "CC"]; ml = [i for i in idx if tr.view2[i] == "MLO"]
        if not cc or not ml: continue
        r, c = linear_sum_assignment(-S[np.ix_(cc, ml)]); pairs += [(cc[a], ml[b]) for a, b in zip(r, c)]
    return pairs

full_cells = [d.index.values for _, d in tr.groupby("lat2")]
SIG = {"convnext": cos(E), "mammo_fm": cos(Xs), "fm+convnext": z(cos(Xs)) + z(cos(E))}
print("== pairing precision (same-label rate, true pairs = 1.0, random ~0.34) ==")
for k, S in SIG.items():
    print(f"{k:12s} same-label={np.mean([y[a] == y[b] for a, b in match(S, full_cells)]):.3f}")

best_S = SIG["fm+convnext"]; pairs = match(best_S, full_cells)
grp = np.arange(len(tr))
for a, b in pairs: grp[b] = a
P = np.zeros((len(y), 3))
for s in range(5):
    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, grp):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
        P[b] += clf.fit(X[a], y[a]).predict_proba(X[b]) / 5
Pf = P.copy()
for a, b in pairs: Pf[[a, b]] = (P[a] + P[b]) / 2
print(f"\nOOF macro F1 (pair-grouped CV) single view = {f1_score(y, P.argmax(1), average='macro'):.3f}")
print(f"OOF macro F1 with pair-averaged probabilities = {f1_score(y, Pf.argmax(1), average='macro'):.3f}")
# upper bound: if pairing were perfect we cannot know on train; approximate with label-consistent pairs only
ok = [(a, b) for a, b in pairs if y[a] == y[b]]
Pu = P.copy()
for a, b in ok: Pu[[a, b]] = (P[a] + P[b]) / 2
print(f"(oracle: averaging only correctly-paired cases) = {f1_score(y, Pu.argmax(1), average='macro'):.3f}")
pd.DataFrame({"image_id": tr.image_id, "pair_fm": grp}).to_csv(OUT / "train_pairs_fm.csv", index=False)
