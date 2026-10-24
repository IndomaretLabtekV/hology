"""EDA 06 - is the test set drawn from the same population as train?

MAE is unforgiving about a global level error, so before tuning a model it is
worth asking whether test listings are systematically richer, longer, or from a
different market mix than training listings.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRICE_FLOOR, SEED, fact_frame, load_raw, repair_text, save_fig, save_table  # noqa: E402

train, test = load_raw()
train["clean"] = train["text"].map(repair_text)
test["clean"] = test["text"].map(repair_text)
y = train["listPrice"].to_numpy(dtype=float)
log_y = np.log(np.clip(y, PRICE_FLOOR, None))

print("=" * 78)
print("06  TRAIN / TEST SHIFT")
print("=" * 78)

corpus = pd.concat([train["clean"], test["clean"]], ignore_index=True)
vec = TfidfVectorizer(sublinear_tf=True, min_df=3, ngram_range=(1, 2),
                      strip_accents="unicode", max_features=300_000)
V = vec.fit_transform(corpus)
Vtr, Vte = V[: len(train)], V[len(train):]

is_test = np.r_[np.zeros(len(train)), np.ones(len(test))]
proba = cross_val_predict(
    LogisticRegression(C=1.0, max_iter=3000), V, is_test,
    cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba",
)[:, 1]
auc = roc_auc_score(is_test, proba)
print(f"adversarial AUC (train vs test) : {auc:.4f}")
print("0.50 = indistinguishable, 0.70+ = a genuinely different population.")

clf = LogisticRegression(C=1.0, max_iter=3000).fit(V, is_test)
terms = np.array(vec.get_feature_names_out())
order = np.argsort(clf.coef_[0])
shift = pd.DataFrame({
    "test_leaning_term": terms[order[-20:]][::-1], "test_coef": clf.coef_[0][order[-20:]][::-1],
    "train_leaning_term": terms[order[:20]], "train_coef": clf.coef_[0][order[:20]],
})
print("\nterms that separate the two splits")
print(shift.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
save_table(shift, "06_adversarial_terms")

print("\n--- surface statistics side by side ---------------------------------")
facts_tr = fact_frame(train["clean"])
facts_te = fact_frame(test["clean"])
rows = []
def add(name, a, b):
    rows.append({"feature": name, "train": float(np.nanmean(a)), "test": float(np.nanmean(b)),
                 "relative_gap": float(np.nanmean(b) / np.nanmean(a) - 1.0)})

add("char_count", train["clean"].str.len(), test["clean"].str.len())
add("word_count", train["clean"].str.split().str.len(), test["clean"].str.split().str.len())
add("digit_count", train["clean"].str.count(r"\d"), test["clean"].str.count(r"\d"))
add("redaction_count", train["clean"].str.count(r"\[Redacted Entity\]"), test["clean"].str.count(r"\[Redacted Entity\]"))
for col in facts_tr.columns:
    add(f"{col}_present", facts_tr[col].notna(), facts_te[col].notna())
    add(f"{col}_value", facts_tr[col], facts_te[col])
surface = pd.DataFrame(rows)
print(surface.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
save_table(surface, "06_surface_statistics")

print("\n--- projected price level of the test set ----------------------------")
oof = np.zeros(len(train))
test_pred = np.zeros(len(test))
for tr_idx, va_idx in KFold(5, shuffle=True, random_state=SEED).split(Vtr):
    model = Ridge(alpha=1.0).fit(Vtr[tr_idx], log_y[tr_idx])
    oof[va_idx] = model.predict(Vtr[va_idx])
    test_pred += model.predict(Vte) / 5.0
print(f"mean predicted log price, train OOF : {oof.mean():.4f}")
print(f"mean predicted log price, test      : {test_pred.mean():.4f}")
print(f"implied test/train price level      : {np.exp(test_pred.mean() - oof.mean()):.3f}x")
proj = pd.DataFrame({
    "quantile": [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
    "train_oof_pred": [float(np.exp(np.quantile(oof, q))) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]],
    "test_pred": [float(np.exp(np.quantile(test_pred, q))) for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]],
})
proj["ratio"] = proj["test_pred"] / proj["train_oof_pred"]
print(proj.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
save_table(proj, "06_projected_test_level")
print("The gap opens only in the upper quantiles: test carries a somewhat")
print("heavier luxury tail than train, so a model that shrinks the tail will be")
print("punished harder on the leaderboard than in cross-validation.")

print("\n--- does the shift correlate with price? -----------------------------")
train_proba = proba[: len(train)]
bins = pd.qcut(train_proba, 10, labels=False)
by_bin = pd.DataFrame({"bin": bins, "price": y}).groupby("bin")["price"].agg(["count", "median", "mean"])
print(by_bin.to_string(float_format=lambda v: f"{v:,.0f}"))
print("If the most test-like training rows are also the priciest, weighting the")
print("training loss toward them is a legitimate way to match the target population.")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].hist(proba[: len(train)], bins=50, alpha=0.6, density=True, label="train")
axes[0].hist(proba[len(train):], bins=50, alpha=0.6, density=True, label="test")
axes[0].set_title(f"adversarial probability (AUC={auc:.3f})")
axes[0].legend()

axes[1].plot(proj["quantile"], proj["ratio"], marker="o")
axes[1].axhline(1.0, color="k", ls="--")
axes[1].set_xlabel("quantile of predicted price")
axes[1].set_ylabel("test / train ratio")
axes[1].set_title("test skews rich in the upper tail")
axes[1].grid(alpha=0.3)

axes[2].plot(by_bin.index, by_bin["median"], marker="o", label="median")
axes[2].plot(by_bin.index, by_bin["mean"], marker="s", label="mean")
axes[2].set_yscale("log")
axes[2].set_xlabel("decile of 'looks like test' probability")
axes[2].set_title("price vs test-likeness")
axes[2].legend()
axes[2].grid(alpha=0.3)
fig.tight_layout()
print("\nfigure:", save_fig(fig, "06_train_test_shift"))
