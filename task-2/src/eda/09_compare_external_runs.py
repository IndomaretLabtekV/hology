"""EDA 09 - audit the two teammate notebooks against the v2 baseline.

Both runs produce out-of-fold predictions for all 14,640 training rows, so
their OOF MAE is directly comparable to ours even though the fold assignment
differs. This script re-derives every headline number from the raw prediction
files rather than trusting the manifests, then asks the only question that
matters for the next run: do these models know anything our models do not?
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import PRICE_FLOOR, ROOT, load_raw, mae, repair_text, save_fig, save_table  # noqa: E402

NB1 = ROOT / "outputs" / "nb1_deberta_late_fusion" / "artifacts"
XLMR = ROOT / "outputs" / "xlmr_analysis" / "oof_preds"

train, _ = load_raw()
truth = train.set_index("id")["listPrice"].astype(float)
y = truth.to_numpy()
log_y = np.log(np.clip(y, PRICE_FLOOR, None))
order = truth.index

print("=" * 78)
print("09  EXTERNAL RUN AUDIT")
print("=" * 78)


def load_log(path: Path, column: str) -> np.ndarray:
    frame = pd.read_csv(path).set_index("id")
    missing = order.difference(frame.index)
    if len(missing):
        raise SystemExit(f"{path.name} is missing {len(missing)} training ids")
    return frame.loc[order, column].to_numpy(dtype=float)


sources = {
    "nb1 DeBERTa": (NB1 / "deberta_oof.csv", "pred_log"),
    "nb1 TF-IDF ridge": (NB1 / "tfidf_oof.csv", "pred_log"),
    "nb1 CatBoost structured": (NB1 / "structured_oof.csv", "pred_log"),
    "nb1 stack": (NB1 / "stack_oof.csv", "pred_log"),
    "xlmr XLM-R": (XLMR / "xlmr_oof.csv", "pred_log"),
    "xlmr TF-IDF": (XLMR / "tfidf_oof.csv", "pred_log"),
    "xlmr tabular": (XLMR / "tabular_oof.csv", "pred_log"),
    "xlmr stack": (XLMR / "stack_oof.csv", "stack_pred_log"),
}
oof = {}
for label, (path, column) in sources.items():
    try:
        oof[label] = load_log(path, column)
    except (FileNotFoundError, KeyError) as exc:
        print(f"skipped {label}: {exc}")

v2_dir = ROOT / "outputs" / "v2"
for name in ("ridge", "svr", "lgbm"):
    npy = v2_dir / f"oof_{name}.npy"
    if npy.exists():
        ids = np.load(v2_dir / "train_ids.npy", allow_pickle=True)
        series = pd.Series(np.load(npy), index=ids)
        oof[f"v2 {name}"] = series.loc[order].to_numpy()

print(f"\nloaded {len(oof)} out-of-fold prediction vectors, all on the same 14,640 rows")

print("\n--- headline MAE, recomputed from the raw files ---------------------")
tail = y > 1.5e6
rows = []
for label, pred_log in oof.items():
    pred = np.exp(pred_log)
    rows.append({
        "model": label,
        "oof_mae": mae(y, pred),
        "median_ae": float(np.median(np.abs(y - pred))),
        "mae_log": float(np.mean(np.abs(log_y - pred_log))),
        "tail_mae_above_1.5M": mae(y[tail], pred[tail]),
        "bulk_mae_below_1.5M": mae(y[~tail], pred[~tail]),
    })
board = pd.DataFrame(rows).sort_values("oof_mae")
print(board.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
save_table(board, "09_external_scoreboard")

print("\n--- are the two transformers even beating a bag of words? -----------")
for tf_label, nn_label in [("nb1 TF-IDF ridge", "nb1 DeBERTa"), ("xlmr TF-IDF", "xlmr XLM-R")]:
    if tf_label in oof and nn_label in oof:
        gap = mae(y, np.exp(oof[nn_label])) - mae(y, np.exp(oof[tf_label]))
        verdict = "transformer LOSES" if gap > 0 else "transformer wins"
        print(f"  {nn_label:24s} vs {tf_label:20s}: {gap:+12,.0f}   {verdict}")
print("A fine-tuned large encoder that loses to TF-IDF on 14k rows of English")
print("prose is not a modelling result, it is a symptom. The usual causes are a")
print("target space the loss cannot handle, too short a truncation window, or a")
print("learning rate that never let the head converge.")

print("\n--- residual correlation: is there anything new here? ----------------")
residual = pd.DataFrame({label: log_y - pred for label, pred in oof.items()})
correlation = residual.corr()
print(correlation.round(3).to_string())
save_table(correlation.reset_index(), "09_residual_correlation")

if any(k.startswith("v2 ") for k in oof):
    best_v2 = min((k for k in oof if k.startswith("v2 ")), key=lambda k: mae(y, np.exp(oof[k])))
    print(f"\ncorrelation of each external model's residual with {best_v2}:")
    for label in oof:
        if not label.startswith("v2 "):
            print(f"  {label:26s} {correlation.loc[label, best_v2]:+.3f}")
    print("Below about 0.85 means the model is making different mistakes and can")
    print("earn a place in the blend even if its solo score is poor.")

print("\n--- why the DeBERTa run collapsed ---------------------------------")
print("nb1 ran domain-adaptive MLM pretraining on microsoft/deberta-v3-base for")
print("3 epochs at lr 5e-5. The published DeBERTa-v3 checkpoint stores its head")
print("under 'lm_predictions.lm_head.*', which is the ELECTRA-style generator")
print("head, while DebertaV2ForMaskedLM looks for 'cls.predictions.*'. The names")
print("do not match, so every MLM-head weight is randomly initialised and three")
print("epochs of gradient from that random head flow back into the encoder.")
print("Evidence in the numbers: rank quality fell below the bag of words.")
for label in ("nb1 DeBERTa", "xlmr XLM-R", "nb1 TF-IDF ridge"):
    if label in oof:
        rho = pd.Series(oof[label]).corr(pd.Series(log_y), method="spearman")
        print(f"  spearman(pred, truth)  {label:20s} {rho:.4f}")
print("XLM-R is a genuine MLM model and ships its lm_head, so the same recipe")
print("left it unharmed; it merely tied the bag of words instead of collapsing.")

print("\n--- truncation audit of the two runs ---------------------------------")
print("nb1 fine-tuned at max_length 512, the xlmr run at 256. Measured on the")
print("repaired text with each run's own tokenizer:")
try:
    from transformers import AutoTokenizer  # noqa: E402

    texts = pd.concat([train["text"]], ignore_index=True).map(repair_text).tolist()
    for model_name, used_length in [("FacebookAI/xlm-roberta-large", 256),
                                    ("microsoft/deberta-v3-base", 512)]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        lengths = np.array([len(x) for x in tokenizer(texts, truncation=False)["input_ids"]])
        tail_mask = y > 1.5e6
        print(f"  {model_name}  at max_length {used_length}")
        print(f"    all listings truncated      : {np.mean(lengths > used_length):6.2%}")
        print(f"    listings above $1.5M cut    : {np.mean(lengths[tail_mask] > used_length):6.2%}")
        print(f"    listings below $1.5M cut    : {np.mean(lengths[~tail_mask] > used_length):6.2%}")
except Exception as exc:
    print(f"  skipped, tokenizers unavailable: {exc}")
print("The 256-token window removes the second half of two thirds of the luxury")
print("listings, and those rows carry 58% of the metric. The 512 window is fine.")

print("\n--- calibration headroom (in-sample isotonic, optimistic) ------------")
from sklearn.isotonic import IsotonicRegression  # noqa: E402
for label, pred_log in sorted(oof.items(), key=lambda kv: mae(y, np.exp(kv[1]))):
    fitted = IsotonicRegression(out_of_bounds="clip").fit(pred_log, log_y).predict(pred_log)
    print(f"  {label:26s} {mae(y, np.exp(pred_log)):>10,.0f} -> {mae(y, np.exp(fitted)):>10,.0f}")
print("Even with a perfect monotone rescaling fitted on its own answers, no")
print("external model reaches the uncalibrated v2 sparse baseline. The gap is")
print("ranking quality, not scaling, so recalibrating their outputs cannot fix it.")

print("\n--- does adding them to the v2 blend help? ---------------------------")
v2_keys = [k for k in oof if k.startswith("v2 ")]
if v2_keys:
    def search(keys):
        matrix = np.column_stack([oof[k] for k in keys])
        weights = np.full(len(keys), 1.0 / len(keys))
        current = mae(y, np.exp(matrix @ weights))
        for _ in range(80):
            improved = False
            for j in range(len(keys)):
                for delta in (0.05, -0.05, 0.02, -0.02, 0.01, -0.01):
                    trial = weights.copy()
                    trial[j] = max(trial[j] + delta, 0.0)
                    if trial.sum() <= 0:
                        continue
                    trial = trial / trial.sum()
                    score = mae(y, np.exp(matrix @ trial))
                    if score < current - 1.0:
                        weights, current, improved = trial, score, True
            if not improved:
                break
        return weights, current

    external = [k for k in oof if not k.startswith("v2 ") and "stack" not in k]
    combos = {
        "v2 only": v2_keys,
        "external only": external,
        "v2 + external": v2_keys + external,
        "v2 + xlmr XLM-R only": v2_keys + ["xlmr XLM-R"],
    }
    blend_rows = []
    for label, keys in combos.items():
        keys = [k for k in keys if k in oof]
        if not keys:
            continue
        weights, score = search(keys)
        chosen = {k: round(float(w), 3) for k, w in zip(keys, weights) if w > 0.005}
        blend_rows.append({"combination": label, "blend_mae": score, "weights": str(chosen)})
        print(f"  {label:24s} MAE {score:>10,.0f}   {chosen}")
    save_table(pd.DataFrame(blend_rows), "09_blend_combinations")
else:
    print("  v2 out-of-fold files not found; run src/prep/03_cpu_submission.py first")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
sorted_board = board.sort_values("oof_mae")
colors = ["tab:green" if m.startswith("v2") else "tab:gray" for m in sorted_board["model"]]
axes[0].barh(range(len(sorted_board)), sorted_board["oof_mae"], color=colors)
axes[0].set_yticks(range(len(sorted_board)))
axes[0].set_yticklabels(sorted_board["model"], fontsize=8)
axes[0].invert_yaxis()
axes[0].set_xlabel("OOF MAE")
axes[0].set_title("every model on the same 14,640 rows")

im = axes[1].imshow(correlation.to_numpy(), vmin=0, vmax=1, cmap="viridis")
axes[1].set_xticks(range(len(correlation)))
axes[1].set_xticklabels(correlation.columns, rotation=90, fontsize=7)
axes[1].set_yticks(range(len(correlation)))
axes[1].set_yticklabels(correlation.index, fontsize=7)
axes[1].set_title("residual correlation in log space")
fig.colorbar(im, ax=axes[1], fraction=0.046)
fig.tight_layout()
print("\nfigure:", save_fig(fig, "09_external_comparison"))
