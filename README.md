# HOLOGY 9.0 HoloMine: Data Mining Competition

Solutions by team **IndomaretLabtekV** for the HoloMine data mining track of HOLOGY 9.0 (2026). There were two preliminary tasks and an 8-hour final. The team finished **4th overall (Juara Harapan 1)**. The final standing combines the private leaderboard with the judged presentation.

| Stage | Task | Metric | Private LB | Rank (private) |
|---|---|---|---|---|
| [Task 1](https://www.kaggle.com/competitions/holomine-breasts-cancer-classification-task-1) | Mammogram classification (Normal / Benign / Malignant) | Macro F1 ↑ | **0.90413** | 5 / 65 |
| [Task 2](https://www.kaggle.com/competitions/holomine-property-price-prediction-from-sales-desc-task-2) | Property price regression from listing text | MAE ↓ | **313,989.54** | 6 / 62 |
| [Final](https://www.kaggle.com/competitions/holomine-paddy-field-segmentation-finals) | Paddy field segmentation on Pleiades tiles | Micro IoU ↑ | **0.64482** | 7 / 10 |
| **Overall** | Leaderboard + presentation | | | **4th (Juara Harapan 1)** |

Hardware: one NVIDIA RTX 5090 (32 GB) rented on Vast.ai for training. EDA scripts run on CPU.

---

## Task 1: Breast Cancer Classification

212 train / 54 test full-field digital mammograms (AISSLab / MDCMI-BC). The test set is patient-disjoint from train.

**Key findings (`task-1/eda/`, 41 scripts)**
- **Format leak.** Normal images are stored as grayscale JPEG q95, abnormal ones as RGB q75. `image.mode` alone predicts Normal with AUC 0.988, and the background JPEG texture still leaks (AUC 0.76). Fix: re-encode everything to q75, zero the background, and never use file metadata as a feature.
- **Cross-view leak.** Each patient contributes a CC and an MLO view with the same label. A 1-NN probe scores 0.780 with plain StratifiedKFold but 0.555 with patient groups. Pairs were recovered with OCR view tags plus Hungarian matching on embeddings (precision 0.93) and used for StratifiedGroupKFold.
- Generic recipes such as a tight crop with a hard mask lowered Benign-vs-Malignant AUC from 0.717 to 0.568, so they were rejected. Robust intensity scaling (white point at the tissue p99.8) and chest-wall flipping were kept.

**Model.** Partial fine-tuning of foundation backbones (Mammo-FM EfficientNet-B5 and DINOv3 ViT) with layer-wise LR decay, a Benign-vs-Malignant auxiliary head, and tissue-crop letterboxing (1344×720 for DINOv3, which raises the share of tokens on tissue from 44% to 63%). Predictions are fused per patient pair and ensembled across folds and seeds.

| Notebook | Description |
|---|---|
| `v1_effnetb3_imagenet_5090` | EfficientNet-B3 ImageNet baseline |
| `v2_…_modeprior`, `v4_…_modeprior` | Teammate baselines (Mammo-FM fine-tune, frozen DINOv3 + MLP). Both use the file-mode prior |
| `v5_dinov3_mammofm_5090` | Leak-free Mammo-FM + DINOv3 probes and fine-tuning, with pair fusion |
| `v6_mammofm_5090` | Mammo-FM with robust normalization, rotation augmentation, and a multi-seed ensemble |
| `v7_dinov3_final_5090` | DINOv3 fine-tune on tissue crops, patient-grouped CV, error analysis |

---

## Task 2: Property Price from Sales Description

14,640 train / 3,659 test listings with a single free-text column. The target `listPrice` ranges from $1 to $80M.

**Key findings (`task-2/reports/findings.md`, `task-2/src/eda/`)**
- MAE is minimised by the conditional median, so all models regress **log price** with an L1-type objective. This one change mattered more than any architecture choice: v1 (ModernBERT on raw dollars) lost to a TF-IDF ridge.
- The 220 listings above $5M (1.5% of rows) carry 31% of the error. The context window is therefore 640 tokens so long luxury listings are not truncated.
- The v2 optimizer skipped three DeBERTa encoder tensors (`rel_embeddings`, encoder `LayerNorm`). Fixing that and training longer gave the v2 → v3 gain.

**Model.** DeBERTa-v3-large fine-tuned on standardized log price (5 folds, 640 tokens, masked mean pooling). It is late-fused with TF-IDF ridge/SVR, and LightGBM on extracted structured facts (area, beds, baths), then blended with isotonic calibration.

| Iteration | OOF MAE | Public LB |
|---|---|---|
| v1: ModernBERT-large, raw price | 470,941 | 419,293 |
| v2: sparse + LightGBM, log target | 326,358 | 408,870 |
| v2: + DeBERTa-v3-large | 277,185 | 329,329 |
| **v3: optimizer fix, 5 epochs** | **267,139** | **310,224** (private 313,990) |

Final notebook: `task-2/notebooks/holomine_v3_rtx5090.ipynb`. The EDA and preprocessing scripts in `task-2/src/` run on CPU in about 15 minutes.

---

## Final: Paddy Field Segmentation

123 labelled train tiles and 129 test tiles (Pleiades RGB, 1024×1024) from Sepatan Timur, Tangerang. Train, public test, and private test come from spatially separate blocks.

**Key findings (`final/eda/`, 18 scripts)**
- Tiles were cut with stride 512, giving 93 overlapping train pairs with identical labels. Validation therefore uses **spatial folds with purging**.
- Every tile is percentile-stretched per channel (about 2% of pixels saturate at 255), so photometric augmentations end with a re-stretch. Haze is applied per mosaic block, because global haze followed by a re-stretch does nothing. Synthetic clouds respect the NoData mask.
- The model is under-confident on test blocks, so the best LB threshold (0.10 to 0.15) sits far below the OOF optimum (0.3 to 0.5).

**Model.** DINOv3 ViT-L/16 **SAT-493M** (satellite-pretrained) encoder with a DPT-style multi-level reassemble + FPN decoder and an RGB stem at stride 4. Training uses BCE + soft-Jaccard loss, AdamW with layer-wise LR decay, and random 768 crops. Inference uses full tiles with D4 TTA (8 views), NoData masking, and a multi-run ensemble.

| File | Description |
|---|---|
| `final/notebooks/paddy_dinov3_final.ipynb` | Presented notebook (ViT-L SAT, 5-fold spatial CV) |
| `final/notebooks/dinov3_v3_7b_5fold.ipynb` | Teammate variant with a frozen DINOv3 ViT-7B SAT backbone |
| `final/presentation.pdf` | Final presentation slides |

---

## Repository Structure

```
task-1/   eda/ (41 audit scripts), notebooks/ (v1–v7), outputs/ (EDA tables and figures), results/ (OOF, figures, submissions)
task-2/   src/eda/, src/prep/, notebooks/, reports/ (findings, figures, tables), outputs/submission.csv
final/    eda/ (18 scripts), notebooks/, outputs/eda/, presentation.pdf
```

## Reproducing

Data and model weights are not tracked. Download each competition's data into its `data/` folder:

```bash
kaggle competitions download -c holomine-breasts-cancer-classification-task-1 -p task-1/data
kaggle competitions download -c holomine-property-price-prediction-from-sales-desc-task-2 -p task-2/data
kaggle competitions download -c holomine-paddy-field-segmentation-finals -p final/data
# unzip each archive in place
```

Notebooks target an RTX 5090 with PyTorch built for CUDA 12.8 or newer, and each one installs its own dependencies in its first cell. DINOv3 weights on Hugging Face are gated: accept the license and set `HF_TOKEN`. EDA scripts run from the task folder, for example `python eda/01_basic_stats.py`.

## Team

IndomaretLabtekV: [athillazaidan](https://www.kaggle.com/athillazaidan), [nicholaswisee](https://www.kaggle.com/nicholaswisee), [riantamaputra](https://www.kaggle.com/riantamaputra)
