"""EDA 02 - what is physically wrong with the text, and does repairing it help.

The descriptions went through a scraper and a redaction pass before reaching us.
This script inventories the damage, measures how often each defect fires in
train vs test, and shows before/after samples so the repair can be eyeballed.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import load_raw, repair_text, save_fig, save_table  # noqa: E402

train, test = load_raw()

DEFECTS = {
    "split_thousands (12, 400)": r"(?<=\d),\s+\d{3}\b",
    "glued_sentence (word.Word)": r"(?<=[a-z0-9])\.(?=[A-Z])",
    "glued_words (no punctuation)": r"\b[a-z]{19,}\b",
    "redacted_entity_token": r"\[Redacted Entity\]",
    "stripped_accent_hyphen (pied--terre)": r"[a-z]--[a-z]",
    "allcaps_shout": r"\b[A-Z]{6,}\b",
    "non_ascii": r"[^\x00-\x7f]",
    "currency_amount": r"\$\s?\d",
    "mls_reference": r"(?i)mls\s*#",
    "no_terminal_period": r"[a-zA-Z0-9]$",
}

print("=" * 78)
print("02  TEXT QUALITY AUDIT")
print("=" * 78)
rows = []
for name, pattern in DEFECTS.items():
    tr_rate = train["text"].str.contains(pattern, regex=True).mean()
    te_rate = test["text"].str.contains(pattern, regex=True).mean()
    rows.append({"defect": name, "train_rate": tr_rate, "test_rate": te_rate, "abs_gap": abs(tr_rate - te_rate)})
audit = pd.DataFrame(rows).sort_values("abs_gap", ascending=False)
print(audit.to_string(index=False, float_format=lambda v: f"{v:7.2%}"))
save_table(audit, "02_defect_rates")

print("\n--- worked examples of the split-thousands defect -------------------")
hit = train["text"].str.contains(r"(?<=\d),\s+\d{3}\b", regex=True)
window = re.compile(r".{0,45}\d,\s+\d{3}.{0,45}")
for text in train.loc[hit, "text"].head(5):
    match = window.search(text)
    if match is None:
        continue
    fragment = match.group(0).strip()
    print(f"  before : ...{fragment}...")
    print(f"  after  : ...{repair_text(fragment)}...")
    print()

print("--- does the repair change the number the parser sees? --------------")
def numbers(text: str) -> list[str]:
    return re.findall(r"\d[\d,\.]*", text)

changed, bigger = 0, []
for text in train.loc[hit, "text"]:
    raw_tokens = numbers(text)
    fixed_tokens = numbers(repair_text(text))
    if raw_tokens != fixed_tokens:
        changed += 1
        if len(bigger) < 5:
            bigger.append((raw_tokens[:6], fixed_tokens[:6]))
print(f"rows whose number stream changes after repair: {changed} / {int(hit.sum())}")
for raw_tokens, fixed_tokens in bigger:
    print(f"  {raw_tokens} -> {fixed_tokens}")

print("\n--- redaction placeholder ------------------------------------------")
tr_red = train["text"].str.count(r"\[Redacted Entity\]")
te_red = test["text"].str.count(r"\[Redacted Entity\]")
print(f"train rows with >=1 placeholder : {(tr_red > 0).mean():.2%}   mean per row {tr_red.mean():.3f}")
print(f"test  rows with >=1 placeholder : {(te_red > 0).mean():.2%}   mean per row {te_red.mean():.3f}")
y = train["listPrice"].to_numpy(dtype=float)
print(f"median price with placeholder   : {np.median(y[tr_red > 0]):,.0f}")
print(f"median price without            : {np.median(y[tr_red == 0]):,.0f}")
print("The placeholder hides the street address, i.e. the single strongest price")
print("driver in real estate. Its test rate is higher, so test rows carry less")
print("location information on average than train rows.")

print("\n--- length profile --------------------------------------------------")
len_rows = []
for split, frame in [("train", train), ("test", test)]:
    chars = frame["text"].str.len()
    words = frame["text"].str.split().str.len()
    len_rows.append({
        "split": split, "n": len(frame),
        "char_mean": chars.mean(), "char_p50": chars.median(),
        "char_p95": chars.quantile(0.95), "char_p99": chars.quantile(0.99), "char_max": chars.max(),
        "word_p95": words.quantile(0.95), "word_p99": words.quantile(0.99), "word_max": words.max(),
    })
lengths = pd.DataFrame(len_rows)
print(lengths.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
save_table(lengths, "02_length_profile")
est_tokens = lengths.loc[1, "char_p99"] / 4.0
print(f"\nchar p99 ~ {lengths.loc[1, 'char_p99']:,.0f} -> roughly {est_tokens:,.0f} wordpiece tokens.")
print("A 768-token window therefore covers ~99% of listings end to end; 1024 is")
print("pure waste of compute, 512 truncates the long luxury descriptions that")
print("carry most of the MAE mass.")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
axes[0].hist(train["text"].str.len(), bins=70, alpha=0.6, density=True, label="train")
axes[0].hist(test["text"].str.len(), bins=70, alpha=0.6, density=True, label="test")
axes[0].axvline(lengths.loc[1, "char_p99"], color="k", ls="--", label="test p99")
axes[0].set_title("character length")
axes[0].legend()

bands = pd.qcut(train["text"].str.len(), 10, labels=False)
grouped = pd.DataFrame({"band": bands, "price": y}).groupby("band")["price"].median()
axes[1].plot(grouped.index, grouped.values, marker="o")
axes[1].set_yscale("log")
axes[1].set_title("median price by text-length decile")
axes[1].set_xlabel("length decile")
axes[1].grid(alpha=0.3)

names = audit["defect"].tolist()
axes[2].barh(np.arange(len(names)) - 0.2, audit["train_rate"], height=0.4, label="train")
axes[2].barh(np.arange(len(names)) + 0.2, audit["test_rate"], height=0.4, label="test")
axes[2].set_yticks(range(len(names)))
axes[2].set_yticklabels(names, fontsize=8)
axes[2].set_title("defect rate, train vs test")
axes[2].legend()
fig.tight_layout()
print("figure:", save_fig(fig, "02_text_quality"))
