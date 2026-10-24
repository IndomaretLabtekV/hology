"""PREP 05 - blend two finished submissions when the OOF is gone.

The v3 instance was destroyed before its out-of-fold predictions were copied
off, so there is nothing left to fit a blend weight against. What survives is
each run's test predictions, recoverable from Kaggle's own submission list.

This script therefore does two different jobs, and keeps them apart:

  1. A diagnostic that needs no labels: how much do the two submissions
     actually disagree. If they barely disagree, blending cannot help and the
     slot is better saved.
  2. A blend at a weight chosen from a stated assumption, not fitted. The
     result is a gamble on one submission slot, and the script says so.

Usage:
    python src/prep/05_blend_submissions.py path/to/submission_v3.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, ROOT, save_table  # noqa: E402

V2_SUB = ROOT / "outputs" / "holomine_v2_rtx5090" / "submission.csv"
OUT = ROOT / "outputs" / "v3"
OUT.mkdir(parents=True, exist_ok=True)

LB = {"v2": 327_089.0, "v3": 312_860.0}

if len(sys.argv) < 2:
    raise SystemExit(
        "berikan path submission v3.\n"
        "ambil dari Kaggle: tab 'My Submissions' -> tombol download pada baris v3"
    )
v3_path = Path(sys.argv[1])

sample = pd.read_csv(DATA / "sample_submission.csv")
v2 = pd.read_csv(V2_SUB).set_index("id")["listPrice"].reindex(sample["id"])
v3 = pd.read_csv(v3_path).set_index("id")["listPrice"].reindex(sample["id"])
if v2.isna().any() or v3.isna().any():
    raise SystemExit("ada id yang tidak cocok antara kedua submisi")

log_v2, log_v3 = np.log(v2.to_numpy()), np.log(v3.to_numpy())

print("=" * 78)
print("PREP 05  BLEND DUA SUBMISI, TANPA OOF")
print("=" * 78)
print("Tidak ada label di sini. Bobot di bawah TIDAK dioptimasi, ia diasumsikan.")

print("\n--- 1. diagnostik: seberapa beda keduanya ----------------------------")
corr = float(np.corrcoef(log_v2, log_v3)[0, 1])
rel = np.abs(v3 - v2) / v2
print(f"korelasi log prediksi        : {corr:.4f}")
print(f"median |selisih relatif|     : {rel.median():.2%}")
print(f"baris berbeda lebih dari 20% : {(rel > 0.20).mean():.1%}")
print(f"baris berbeda lebih dari 50% : {(rel > 0.50).mean():.1%}")
print(f"std selisih log              : {np.std(log_v3 - log_v2):.4f}")

if corr > 0.99:
    verdict = ("Keduanya praktis identik. Blending tidak akan menggerakkan skor; "
               "simpan slotnya.")
elif corr > 0.97:
    verdict = ("Beda tipis. Blending mungkin menggeser beberapa ribu, arahnya "
               "tidak bisa dipastikan tanpa label.")
else:
    verdict = ("Cukup berbeda untuk membuat blending masuk akal, meski tetap "
               "tanpa jaminan.")
print(f"\n{verdict}")

print("\n--- 2. bobot, dari asumsi yang dinyatakan ----------------------------")
print("Asumsi: error kedua model di ruang log punya skala sebanding dengan MAE")
print("leaderboard-nya, dan korelasi residual mendekati korelasi prediksi.")
print("Dengan itu, bobot yang meminimalkan varians gabungan adalah")
print("  w2 = (s3^2 - rho*s2*s3) / (s2^2 + s3^2 - 2*rho*s2*s3)")
s2, s3 = LB["v2"], LB["v3"]
denom = s2 ** 2 + s3 ** 2 - 2 * corr * s2 * s3
w_v2 = (s3 ** 2 - corr * s2 * s3) / denom if denom > 0 else 0.0
w_v2 = float(np.clip(w_v2, 0.0, 0.5))
print(f"\n  LB v2 {s2:,.0f}   LB v3 {s3:,.0f}   rho {corr:.4f}")
print(f"  -> bobot v2 = {w_v2:.3f}, bobot v3 = {1 - w_v2:.3f}")
print("\nIni formula untuk kuadrat kesalahan, bukan MAE, dan korelasi prediksi")
print("bukan korelasi residual. Perlakukan sebagai titik awal, bukan optimum.")

rows = []
for weight in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    blended = np.exp((1 - weight) * log_v3 + weight * log_v2)
    rows.append({
        "bobot_v2": weight,
        "median_pred": float(np.median(blended)),
        "q95": float(np.quantile(blended, 0.95)),
        "q99": float(np.quantile(blended, 0.99)),
        "median_geser_dari_v3": float(np.median(np.abs(blended - v3.to_numpy()))),
    })
grid = pd.DataFrame(rows)
print("\nefek tiap bobot pada sebaran prediksi")
print(grid.to_string(index=False, float_format=lambda v: f"{v:,.2f}" if v < 1 else f"{v:,.0f}"))
save_table(grid, "prep05_blend_grid")

chosen = round(w_v2, 2)
price = np.exp((1 - chosen) * log_v3 + chosen * log_v2)
submission = pd.DataFrame({"id": sample["id"], "listPrice": price})
assert submission["listPrice"].notna().all() and (submission["listPrice"] > 0).all()
path = OUT / f"submission_blend_v2w{int(chosen * 100):02d}.csv"
submission.to_csv(path, index=False)
print(f"\nditulis ke {path}   (bobot v2 = {chosen:.2f})")

train = pd.read_csv(DATA / "train.csv")
y = train["listPrice"].to_numpy(dtype=float)
comparison = pd.DataFrame({
    "quantile": [0.05, 0.5, 0.95, 0.99],
    "train": [np.quantile(y, q) for q in [0.05, 0.5, 0.95, 0.99]],
    "v3": [np.quantile(v3, q) for q in [0.05, 0.5, 0.95, 0.99]],
    "blend": [np.quantile(price, q) for q in [0.05, 0.5, 0.95, 0.99]],
})
print(comparison.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))
print("\nKalau sebaran blend nyaris sama dengan v3, gain-nya juga akan nyaris nol.")
