# HoloMine Task 2 - EDA findings and what they imply

Every number below comes from a script in `src/`. Re-run any of them to reproduce.

## 1. The metric and the target disagree about scale

`src/eda/01_target_geometry.py`

| statistic | value |
| --- | --- |
| median | 499,900 |
| mean | 839,073 |
| max | 80,000,000 |
| skewness, raw | 15.29 |
| skewness, log | -0.22 |
| MAE of constant median | 550,252 |

MAE is minimised by the **conditional median**, not the conditional mean. In
the constant-predictor test the median beats the mean by 101,416 dollars.

Where the error mass sits, measured against the median baseline:

| price band | rows | share of rows | share of MAE |
| --- | --- | --- | --- |
| < 500k | 7,386 | 50.4% | 19.2% |
| 500k - 1.5M | 5,913 | 40.4% | 22.3% |
| 1.5M - 5M | 1,121 | 7.7% | 27.5% |
| > 5M | 220 | 1.5% | 31.0% |

The 500 most expensive listings (3.4% of rows) carry 42% of the total error.

**Implication.** Optimise the tail or do not bother. Middle-of-the-market
accuracy is nearly free and nearly worthless.

## 2. Log space is worth more than any architecture

`src/eda/07_target_transform_and_loss.py`, identical folds and identical model:

| target space | OOF MAE |
| --- | --- |
| raw dollars | 514,990 |
| price / 1e6 | 514,990 |
| sqrt dollars | 363,684 |
| log dollars | **356,188** |

A 30.8% improvement from one line of code. The v1 notebook regressed
`price / 1e6` with a Huber loss and scored 632,642 with ModernBERT-large,
worse than a TF-IDF ridge. The architecture was never the problem.

Because `exp` is monotone, `median(price | text) = exp(median(log price | text))`.
So the correct recipe is: regress log price with an L1-like objective, then
exponentiate. An L2 objective in log space returns the conditional mean of log
price, which exponentiates to the geometric mean and sits below the median.

## 3. The text is corrupted in two systematic ways

`src/eda/02_text_quality_audit.py`

| defect | train | test |
| --- | --- | --- |
| thousands separator split into ", " | 4.41% | 5.44% |
| sentence period lost its space | 11.41% | 7.38% |
| `[Redacted Entity]` placeholder | 14.02% | 17.76% |

`"this 2, 557-square-foot residence"` is really 2557 square feet. The number
stream changes on **all 646** affected train rows once repaired, and the parsed
`sqft` value changes on 345 rows, median 547 to 2,585.

**Verified, not assumed** (`src/prep/02_verify_dataset.py`): on a bag-of-ngrams
model the repair is neutral (356,158 raw vs 356,183 repaired, against a
fold-to-fold noise of 25,000). It is kept because the fact parser and the
subword tokenizer both read magnitudes and a bag of ngrams does not.

## 4. Preprocessing steps that are standard and wrong here

Same folds, same model, word 1-2gram TF-IDF ridge on log price:

| variant | OOF MAE | verdict |
| --- | --- | --- |
| model_text | 356,183 | shipped |
| digits stripped | 355,919 | neutral, kept for the transformer |
| casing preserved (`lowercase=False`) | 360,940 | **loses 4,757** |
| proper nouns stripped | 374,573 | **loses 18,390** |
| + extracted facts, full weight | 357,179 | **loses 996** |
| + extracted facts, weight 0.05 | 355,800 | marginal |

Two lessons. Proper nouns are the only location evidence left after the address
redaction, so deleting them is expensive. And concatenating 48 unregularised
dense columns onto 300k sparse ones drowns them under a single ridge alpha; the
extracted facts belong in a tree model, not in the linear text model.

## 5. The dataset is two markets wearing one hat

`src/eda/04_lexical_and_location_signal.py`

| place | rows | median price | vs global |
| --- | --- | --- | --- |
| Central Park | 125 | 1,795,000 | 3.59x |
| Manhattan | 276 | 1,025,000 | 2.05x |
| Lake Oswego | 87 | 935,000 | 1.87x |
| Buffalo | 40 | 212,450 | 0.42x |
| Rochester | 26 | 154,950 | 0.31x |
| Klamath Falls | 46 | 77,500 | 0.16x |

The corpus is New York State plus Oregon: NYC luxury, upstate New York rust
belt, and Oregon. The spread between the cheapest and most expensive recognised
market is 23x, far larger than any property-attribute effect. 58.3% of listings
name at least one place and 74.2% of place mentions in test also occur in train.

Concept probes, median price with the keyword over median without:

| concept | ratio |
| --- | --- |
| Fifth Avenue | 9.99x |
| wine cellar | 4.30x |
| Central Park | 3.00x |
| chef's kitchen | 2.68x |
| auction | 0.42x |
| vacant lot | 0.36x |
| manufactured / mobile home | 0.49x |

## 6. Structured facts are recoverable but sparse

`src/eda/03_structured_facts.py`

| fact | coverage | Spearman vs log price |
| --- | --- | --- |
| beds | 52.2% | +0.237 |
| baths | 39.3% | +0.495 |
| acres | 21.8% | +0.190 |
| sqft | 20.2% | +0.447 |
| year_built | 5.9% | -0.114 |

The negative sign on `year_built` survived an anchored re-parse (`built in
YYYY` only), so it is real: historic New York housing stock outprices new
construction in cheap markets. Implied price per square foot spans 122 to 1,176
between the 10th and 90th percentile, which is the location effect that area
alone cannot explain.

Mentioning a fact is itself a signal, independent of its value: listings that
state their square footage have a 1.33x higher median price than listings that
do not. Every fact therefore produces two features, value and missing-flag.

## 7. Duplicates are not the leak people expect

`src/eda/05_duplication_and_cv.py`

- 464 intra-train pairs above cosine 0.85, covering 336 rows (2.30%).
- Plain KFold 360,527 vs GroupKFold 361,460. The "leak" is worth -933 dollars,
  which is noise. Grouping is kept as cheap insurance, not as a fix.
- The real CV problem is tail placement. Fold MAE standard deviation is 30,256
  under GroupKFold and 24,785 under StratifiedGroupKFold on log-price bins.

**Implication.** Any CV delta below roughly 15,000 dollars is noise. Do not
chase it.

Eight test rows have a train twin at cosine >= 0.99 and 35 at >= 0.95. Copying
the train price for those is free accuracy.

## 8. Test skews richer than train

`src/eda/06_train_test_shift.py`

Adversarial AUC 0.5969, so the splits are close but not identical. Projecting a
model onto both sides:

| quantile | train OOF | test | ratio |
| --- | --- | --- | --- |
| 0.50 | 477,059 | 485,914 | 1.02 |
| 0.90 | 1,130,536 | 1,310,907 | 1.16 |
| 0.99 | 3,425,582 | 4,466,363 | 1.30 |

Training rows that look most like test have mean price 1,317,662 against
586,712 for those that look least like it, a 2.2x gradient.

**Implication.** Leaderboard MAE will land above CV MAE, and a model that
shrinks the tail is punished harder on the leaderboard than in validation.

## 9. Where the remaining error lives, and the floor

`src/eda/08_error_anatomy_and_ceiling.py`, ridge on log price:

| true price band | rows | median pred / true | share of MAE |
| --- | --- | --- | --- |
| < 150k | 1,352 | 2.147 | 2.8% |
| 300k - 500k | 3,674 | 1.019 | 8.7% |
| 800k - 1.5M | 2,279 | 0.809 | 16.2% |
| 1.5M - 5M | 1,121 | 0.663 | 22.2% |
| > 5M | 220 | 0.481 | 32.3% |

The model is unbiased in the middle and predicts less than half the true price
at the top. 25 rows out of 14,640 carry 13.6% of all absolute error.

The irreducible floor: pairs of listings whose text is effectively identical
still disagree on price by a mean of 122,705 dollars and a median of 10,000.
No text-only model can beat that on those rows.

Feature ladder, same folds:

| features | OOF MAE |
| --- | --- |
| word TF-IDF | 356,188 |
| word + char TF-IDF | 348,599 |
| word + char + extracted facts (ridge) | 353,945 |

## 10. Un-shrinking the tail is the trap

The shrinkage in section 9 looks like a bias waiting to be corrected. It is not.
Cross-fitted calibration of a LinearSVR baseline (348,449 uncalibrated):

| calibration | top-band median ratio | OOF MAE |
| --- | --- | --- |
| none | 0.622 | 348,449 |
| isotonic | 0.632 | 347,681 |
| isotonic then linear stretch | - | **345,952** |
| rank matching onto the train marginal | 0.802 | **389,219** |

Rank matching fixes the ratio almost perfectly and costs 41,000 dollars of MAE.
The reason is that MAE asks for the conditional median, and the conditional
median of an uncertain prediction *should* sit near the centre. Inflating it
converts bias into variance, and variance is what MAE charges for. The
shrinkage is the model being correct about its own uncertainty.

**Implication.** Calibration is worth about 2,500 dollars, not 100,000. The
only real way to lift the tail is to reduce the uncertainty, which means a model
that actually reads the text. That is the transformer's job, not the
post-processor's.

## 11. What the full CPU-only stack reaches

Same folds, log target throughout:

| model | OOF MAE |
| --- | --- |
| ridge, word + char TF-IDF | 349,135 |
| LinearSVR, word + char TF-IDF | 348,795 |
| LightGBM on extracted facts + 192 SVD components | 348,912 |
| blend (0.51 SVR + 0.48 LightGBM + 0.01 ridge) | 333,127 |
| blend + cross-fitted isotonic | 327,107 |
| blend + isotonic + stretch x0.98, then twin override | **326,358** |

Three weak-ish models blend to 15,000 below the best single one, because their
residuals disagree. For reference, v1 scored 470,941. The floor set by
near-duplicate disagreement is roughly 120,000 on the affected rows.

## What this dictates for the model

1. Regress log price, L1-like loss, exponentiate. Non-negotiable.
2. Keep casing and digits; keep proper nouns. Repair the split thousands.
3. StratifiedGroupKFold on log-price bins, 5 folds.
4. Feed the extracted facts to a tree, not to the linear text model.
5. Calibrate afterwards, but expect small change. Isotonic plus a linear
   stretch, both cross-fitted, is worth about 2,500 dollars. Rank matching
   looks right and costs 41,000; do not use it.
6. Override the handful of test rows that have a near-exact train twin.
7. Expect the leaderboard to read worse than CV, because test is richer.

## Referensi

Tidak ada paper publik yang memakai dataset ini; ia dirakit ulang untuk HOLOGY
dari listing properti New York dan Oregon. Dua rujukan terdekat:

- Shi, Mueller, Erickson, Li, Smola. *Benchmarking Multimodal AutoML for Tabular
  Data with Text Fields.* NeurIPS 2021 Datasets and Benchmarks.
  <https://arxiv.org/abs/2111.02705> - membandingkan 18 tabel multimodal dengan
  kolom teks. Kesimpulannya: yang menang adalah **stack ensemble antara
  transformer multimodal dan model pohon**, bukan salah satunya sendirian. Itu
  persis susunan yang dipakai di sini, yaitu DeBERTa ditambah LightGBM ditambah
  model linear sparse, digabung pada OOF.
- Kaggle Mercari Price Suggestion Challenge - kompetisi regresi harga dari teks
  yang paling mirip secara bentuk. Solusi teratas seragam pada satu hal: target
  dilatih di ruang log, bukan di ruang mata uang.
- Wilie et al. dan literatur encoder 2025 hingga 2026 (`arxiv.org/html/2504.08716v1`)
  membandingkan ModernBERT dan DeBERTaV3 pada beban kerja yang sama. DeBERTaV3
  tetap unggul pada efisiensi sampel dan skor akhir, sementara ModernBERT unggul
  pada kecepatan dan konteks panjang. Karena korpus ini pendek, median 185 token,
  DeBERTa-v3-large dijadikan backbone utama dan ModernBERT-large dipakai sebagai
  sumber keragaman untuk ensemble.

## 12. Audit of the two teammate runs

`src/eda/09_compare_external_runs.py`. Every number below is recomputed from the
raw prediction files, on the same 14,640 training rows.

| model | OOF MAE | tail MAE (>$1.5M) | bulk MAE |
| --- | --- | --- | --- |
| v2 blend + isotonic + XLM-R | **321,199** | - | - |
| v2 blend + isotonic | 327,246 | - | - |
| xlmr stack (catboost, nested CV) | 344,282 | 2,089,193 | 172,226 |
| nb1 stack | 359,772 | 2,173,018 | 180,978 |
| xlmr TF-IDF | 383,760 | 2,498,072 | 175,280 |
| nb1 TF-IDF ridge | 390,010 | 2,560,098 | 176,031 |
| xlmr XLM-R | 400,258 | 2,166,975 | 226,052 |
| nb1 CatBoost structured | 436,769 | 2,585,926 | 224,853 |
| nb1 DeBERTa | 594,575 | 2,226,460 | 433,665 |

Both stacks are honest: the xlmr meta-model is fitted with its own nested KFold,
so 344,282 is not an in-sample number. The problem is in the base models.

### Three verified execution faults

**1. The DAPT step destroyed the DeBERTa backbone.** nb1 ran 3 epochs of masked
language modelling on `microsoft/deberta-v3-base` at lr 5e-5. That checkpoint
stores its head as `lm_predictions.lm_head.*`, the ELECTRA-style generator head,
while `DebertaV2ForMaskedLM` looks for `cls.predictions.*`. Loading it reports
every head tensor as UNEXPECTED or MISSING, so the MLM head is randomly
initialised and three epochs of gradient from it flow back into the encoder.

The damage shows up in rank quality, which no calibration can repair:

| model | spearman(pred, truth) |
| --- | --- |
| nb1 TF-IDF ridge | 0.772 |
| xlmr XLM-R | 0.770 |
| nb1 DeBERTa | 0.645 |

XLM-R is a genuine MLM model and ships its `lm_head`, so the identical recipe
left it unharmed. It merely tied the bag of words instead of collapsing.

**2. `max_length=256` in the XLM-R run.** Measured with XLM-R's own tokenizer on
the repaired text:

| window | all listings truncated | listings above $1.5M truncated |
| --- | --- | --- |
| 256 | 34.68% | **67.12%** |
| 512 | 3.73% | - |
| 640 | 1.40% | - |

Two thirds of the luxury listings lost their second half, and those rows carry
58% of the metric. nb1's 512 window was fine.

**3. Over-regularised ridge.** nb1 used `alpha=12`, the xlmr run `alpha=10`,
against `alpha=1` here. The effect is visible in the prediction spread: nb1's
TF-IDF has a log-prediction standard deviation of 0.619 against a target
standard deviation of 0.983. The model is shrunk flat and the tail goes with it.

Minor: both runs strip thousands separators with `(?<=\d),(?=\d)`, which
matches `1,850` but not the `"2, 557"` corruption documented in section 3. Their
`sqft` still parses as 557 on those rows.

### What is worth taking from them

| combination | blend | + cross-fitted isotonic | + stretch |
| --- | --- | --- | --- |
| v2 (ridge + svr + lgbm) | 333,129 | 327,246 | 327,503 |
| v2 + xlmr XLM-R | 331,249 | 322,341 | **321,199** |

XLM-R earns weight 0.153 and 6,047 dollars of MAE after calibration. Adding
their CatBoost and tabular models on top moves nothing. The gain is smaller than
the 24,785 fold-to-fold standard deviation, so it is a cheap directional win
rather than a result, but it costs nothing: their test predictions cover all
3,659 rows and are keyed by id.

Note the sequencing trap. Measured against the sparse pair alone, XLM-R looked
worth 14,058. Once LightGBM is in the blend, which already carries 192 SVD
components of the text, the same model is worth 1,880 before calibration. A
blend gain is only meaningful against the configuration that actually ships.

## 13. The leaderboard agrees with cross-validation

`src/eda/10_cv_lb_calibration.py`. Five submissions now have both scores.

| submission | OOF MAE | public LB | LB - OOF | LB / OOF |
| --- | --- | --- | --- | --- |
| v1 ModernBERT, dollar target | 470,941 | 567,900 | +96,959 | 1.206 |
| nb1 stack | 359,772 | 434,473 | +74,701 | 1.208 |
| xlmr stack | 344,282 | 415,954 | +71,672 | 1.208 |
| v2 sparse + lgbm | 326,358 | 412,964 | +86,606 | 1.265 |
| v2 sparse + lgbm + xlmr | 321,199 | 402,234 | +81,035 | 1.252 |

Spearman between the two columns is **1.000**, with no inversion anywhere.
`LB = 41,091 + 1.1128 x OOF` fits with R2 = 0.9887.

Two things follow. Cross-validation can be trusted to pick between models, so
there is no reason to spend submissions probing. And the leaderboard sits about
82,000 above CV, which is exactly what section 8 predicted from the heavier
luxury tail in test; the offset is a property of the split, not of any model.

Projection, with the three estimators disagreeing by roughly 12,000:

| OOF | LB (linear) | LB (ratio) | LB (offset) |
| --- | --- | --- | --- |
| 300,000 | 374,920 | 368,361 | 382,195 |
| 280,000 | 352,665 | 343,804 | 362,195 |
| 260,000 | 330,410 | 319,247 | 342,195 |

## 14. The v2 transformer is working

Per-fold MAE on the shared fold assignment:

| model | overall | fold0 | fold1 | fold2 | fold3 | fold4 |
| --- | --- | --- | --- | --- | --- | --- |
| ridge | 349,135 | 366,759 | 344,146 | 349,053 | 375,009 | 310,710 |
| svr | 348,795 | 365,760 | 343,703 | 352,132 | 372,180 | 310,199 |
| lgbm | 348,912 | 366,400 | 339,919 | 341,289 | 384,268 | 312,685 |
| xlmr (theirs) | 400,258 | 425,914 | 399,339 | 400,990 | 415,037 | 360,007 |
| nb1 DeBERTa | 594,575 | 601,188 | 609,720 | 601,033 | 615,210 | 545,726 |

Fold difficulty, as a multiple of each model's own overall score: fold 0 is
1.0533, fold 3 is 1.0699, fold 4 is 0.8937. A number from one fold cannot be
read without this table.

**v2 DeBERTa-v3-large scored 297,517 on fold 0**, which is the second-hardest
fold. The best existing single model on that same fold is 365,760, so the
transformer is ahead by **68,243** on like-for-like data. Rescaled by fold-0
difficulty it projects to roughly **282,000 OOF**.

Caveats that keep this a signal rather than a result: the checkpoint was chosen
on the same validation fold, which flatters it by a few thousand, and one fold
is one fold. But the direction is unambiguous, and it is the first time on this
dataset that a transformer has beaten the bag of words rather than losing to it.
The difference from the two teammate runs is the absence of a DAPT step, a 640
token window instead of 256, and an L1 objective on log price.

## 15. The v2 optimizer never received three parameters

`src/eda/11_optimizer_clip_bug.py`

`build_optimizer` collected `embeddings + encoder.layer`. DeBERTa keeps three
tensors directly on the encoder:

| parameter | shape (base) | role |
| --- | --- | --- |
| `encoder.rel_embeddings.weight` | (512, 768) | the relative position table behind disentangled attention |
| `encoder.LayerNorm.weight` | (768,) | encoder output norm |
| `encoder.LayerNorm.bias` | (768,) | encoder output norm |

That is 0.21% of parameters by count, which understates it twice over. First,
`rel_embeddings` is the component that makes DeBERTa's attention disentangled,
so freezing it pins the positional prior. Second, and worse, `zero_grad` only
clears parameters inside the optimizer, so these three accumulate gradient that
nothing resets, and `clip_grad_norm_(model.parameters(), 1.0)` includes them.

The simulation does **not** show an unbounded blow-up, and the reason is
instructive: `clip_grad_norm_` rescales every `.grad` in place, including the
orphaned ones, so the leak reaches an equilibrium. What it leaves behind is a
permanent tax:

| | steady-state clip coefficient |
| --- | --- |
| optimizer misses a parameter | 0.684 |
| full coverage | 0.970 |
| effective learning rate retained | **70.5%** |

Roughly 30% of the requested learning rate was discarded at every step, for the
whole run. This is consistent with all five folds scoring their best at the
final evaluation while still improving by 8,500 during epoch 3: the model never
converged, it ran out of budget.

The fix is to build the optimizer from `named_parameters()`, assign the
layer-wise rate by looking up each parameter's block, and assert that every
trainable parameter is registered exactly once.

## 16. Calibration and segment corrections are exhausted

`src/eda/12_v2_residual_anatomy.py`, on the real v2 out-of-fold predictions
(blend 277,185, DeBERTa alone 279,714).

Cut by true price, the predictions look badly biased: median ratio 1.711 below
150k and 0.665 above 5M. That table is a trap. Binning on the true value selects
on the noise, so even a perfect conditional median shows the pattern. The cut a
model can act on is by its own prediction:

| prediction decile | median pred / median true |
| --- | --- |
| 0 | 1.049 |
| 4 | 0.997 |
| 9 | 0.929 |

Largest deviation from 1.0 across all ten deciles is **0.071**. There is no
systematic bias left. Every global stretch tested makes MAE worse (1.05 costs
3,569; 1.20 costs 64,194), and the audit's own multiplier search saved 14
dollars.

Segment corrections fail the same test. Seven segments recognisable from the
text at inference time:

| segment | rows | share of error | median pred / true |
| --- | --- | --- | --- |
| land / ranch | 4,783 | 36.5% | 0.998 |
| none of the above | 6,530 | 27.9% | 1.004 |
| NYC apartment/townhouse | 998 | 24.0% | 0.987 |
| multifamily / portfolio | 2,288 | 17.5% | 1.017 |
| new construction | 473 | 4.2% | 0.989 |
| distressed | 770 | 2.9% | 0.968 |
| manufactured | 168 | 0.8% | 0.945 |

NYC does carry 24% of the error on 6.8% of rows, but its median ratio is 0.987.
The error there is variance, not bias, and a segment multiplier has nothing to
multiply. Over-prediction by more than 2x affects 844 rows and under-prediction
by more than 2x affects 755; a correction that lifted expensive-sounding
listings would fix the second group and break the first.

**Conclusion: post-processing is finished. Any further gain must come from the
model knowing more, which means training.**

## 17. What v3 changes, and what it deliberately does not

Locked, because sections 15 and 16 say the remaining gain is in training:

| change | evidence |
| --- | --- |
| optimizer built from `named_parameters()` with a coverage assertion | section 15, 30% of the learning rate was being discarded |
| 5 epochs with early stopping on 3 stale evaluations | all five v2 folds peaked at the final evaluation, still falling 8,500 per epoch |
| ModernBERT dropped | weight 0.0 in the v2 blend, 34 minutes of GPU time returned to DeBERTa |
| per-fold predictions written to disk as each fold ends | v2 kept `best_state` in RAM only; an interrupted run lost everything |
| per-fold wall-clock budget, best checkpoint still used | a rented GPU with a deadline needs a graceful stop, not a crash |

Deliberately not changed, with the reason:

| not done | why |
| --- | --- |
| numeric features into the regression head | plausible, untested; a final run is the wrong place for a new architecture |
| raw-dollar MAE objective | same, plus v1 already showed dollar space is hard to train |
| segment-aware corrections | section 16, every segment's median ratio is within 0.06 of 1.0 |
| further calibration | section 16, largest deviation across prediction deciles is 0.071 |
| parser fixes for `1/2 acre`, `41+ acres` | real bugs, but they feed LightGBM, which carries weight 0.0095 in the blend. Fixing them cannot move the score this run. |
| longer context | truncation at 640 tokens is already 0.55% |

The parser point deserves emphasis because it is the kind of work that feels
productive and is not. LightGBM's blend weight is under one percent. A perfect
parser multiplied by 0.0095 is still nothing. It belongs in a version where the
numeric features reach the transformer, not in this one.
