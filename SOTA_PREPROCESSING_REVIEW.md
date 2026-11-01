# SOTA-Informed Preprocessing and Model Review

**Status:** Stage 1 research review for user approval  
**Project:** IEEE-CIS fraud scoring with delayed-feedback contextual bandit  
**Scope:** Research and recommendations only. No preprocessing code changed, no Kaggle notebooks created, and no new experiments run in this stage.

## Executive summary

The strongest directly relevant competition evidence supports three supervised models—**XGBoost, CatBoost, and LightGBM**—combined with transaction-entity features, including a card/address/day UID and group statistics. The winning solution also used time-aware validation and prediction blending. Its public/private Kaggle AUC scores and local validation AUC are useful references, but are not directly comparable to this project's PR-AUC, chronological split, calibrated decisions, or bandit costs.

The winning competition feature engineering is a useful starting point, but its global group counts/aggregates and use of train/test rows are not safe to copy unchanged into an online chronological replay. The project already has an original, causal implementation: history features use strictly earlier `TransactionDT` values, same-time rows are grouped together, and feature construction does not read `isFraud`. The proposed expansion should compare this control, UID, full causal transaction features, and a separate identity-joined recipe.

**Source format note:** I found the first-place Kaggle notebook, official competition write-ups/discussions, and research papers; I did not find a first-place PowerPoint deck. This review uses those primary materials rather than treating unrelated slide decks as SOTA evidence.

**Approval gate:** This document is the first deliverable. Do not apply these recommendations to the project, alter preprocessing, create experiment notebooks, or launch Kaggle experiments until the user approves the selected recipe/model scope.

## 1. Reference evidence and limits

| Reference | Reported approach/result | What it supports | Comparability limit |
|---|---|---|---|
| IEEE-CIS first-place solution, summarized by NVIDIA | XGBoost, CatBoost, and LightGBM ensemble; UID and group-aggregation engineering; time-aware validation. UID features raised the author's local validation AUC to **0.9472**. Final competition result: **0.9677 public AUC / 0.9459 private AUC**. | Test the three GBDT families, causal UID/group features, and a validation-only rank/score blend. | Competition score is ROC-AUC on Kaggle's public/private test partitions. It is not PR-AUC, cost, or contextual-bandit performance. The local validation protocol differs from this project's 48/12/40 chronological setup. [NVIDIA winner overview](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-develop-a-winning-kaggle-solution/) |
| First-place RAPIDS feature-engineering notebook | Generates roughly **262 features**; reports about **5 minutes** to read/process one million rows with Pandas CPU versus **20 seconds** with RAPIDS GPU. It uses group counts/statistics, combined card/address/email keys, UID, amount cents, and identity data. | Establishes feature families and shows GPU acceleration can help large feature-generation workloads. | Hardware, software versions, selected columns, and computation differ. Our measured baseline is 130 seconds for 590,540 rows and 423 artifact columns on Kaggle CPU; direct speed comparisons are not valid. [Kaggle notebook](https://www.kaggle.com/code/cdeotte/rapids-feature-engineering-fraud-0-96) |
| IEEE-CIS 10th-place solution write-up | Uses `day - D1` UID logic, UID encodings, and time-window transaction counts for UID groups; documents overfitting controls for rare IDs. | Supports testing UID-derived history and windowed group activity; emphasizes unseen/rare-entity robustness. | Competition-specific CV and labels differ. The time-window features need prior-only calculation for this project. [Kaggle write-up](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/m5-10th-solution-and-code) |
| Recent chronological IEEE-CIS study | Reports CatBoost PR-AUC **0.489 ± 0.001** and XGBoost **0.478 ± 0.003** in a 70/15/15 chronological evaluation. | Provides a closer PR-AUC reference than leaderboard ROC-AUC, and supports keeping CatBoost/XGBoost in the comparison. | Still uses a different split, feature handling, and study objective; it is not a direct benchmark or a universal state-of-the-art score. [Study](https://www.mdpi.com/2076-3417/16/12/5809) |

**Current project reference:** LightGBM validation PR-AUC **0.5310**, ROC-AUC **0.8792**; simulation-period ROC-AUC **0.8674**, PR-AUC **0.4854**. The validation ROC-AUC is **0.0680 below** the winner's reported local validation ROC-AUC, but that difference is descriptive only because the splits and validation designs differ. The current project also optimizes a cost-sensitive policy and delayed feedback, which the competition leaderboard does not measure. Results/settings are recorded in [`RUN_LOG.md`](RUN_LOG.md).

## 2. Current project preprocessing audit

The current pipeline reads the transaction file, sorts chronologically, and generates the feature artifact once for model reuse. The artifact has **590,540 rows × 423 columns** and **20 categorical columns**. It retains the raw transaction fields except identifiers/target/time fields excluded from model input and adds engineered columns.

| Existing block | Current content | Leakage/fit boundary |
|---|---|---|
| Control | Transaction day/hour, log amount, prior counts and amount means for card, card/address, and email groups, plus the original transaction features. | Counts and means use strictly earlier timestamps; ties do not update one another. |
| Frequency | Prior counts for additional card/address and card/address/email keys. | No target encoding; no future rows contribute. |
| UID | `card1 + addr1 + floor(transaction_day - D1)` UID prior count. | UID is used for history features, not as a target-derived label. |
| Aggregates | Prior `TransactionAmt` and `D9` means/stds for selected card/address/email/UID groups. | Strictly prior-only; same-time group values are excluded. |
| Amount parts | Dollar and cents components. | Derived from the current transaction amount only. |
| Identity/device | Not joined in the current experiment. | To be tested as a separate ablation using `TransactionID`. |

The implementation is local project code inspired by published competition ideas; it is not a copied Kaggle notebook. Existing feature construction is in `src/fraudlab/features.py`; recipe selection is supported by `feature_columns_for_recipe`. Existing model preparation already distinguishes native categorical handling for LightGBM/CatBoost from XGBoost's low-cardinality one-hot approach. The current artifact and measured run are described in `RUN_LOG.md`.

## 3. Candidate preprocessing choices

| Candidate | Expected benefit / evidence | Leakage risks and required safe adaptation | Model fit | Compute / efficiency estimate |
|---|---|---|---|---|
| **R0: Current control** | Strong reference that preserves the current transaction features and simpler history signals. | Keep strict timestamp ordering and train-fitted encodings/imputation. | LightGBM, CatBoost, XGBoost; one-class models need numeric conversion/scaling. | Already implemented; artifact generation measured at **~130 seconds** on Kaggle CPU. |
| **R1: Control + UID history** | UID from `card1`, `addr1`, and `D1`-derived day was a major feature in the first-place solution; local validation AUC reported as 0.9472. | Competition UID aggregates may mix future train/test rows. Use only prior transactions, do not use UID-level target/fraud averages, and evaluate rare/unseen UIDs. | All three GBDTs; particularly natural with CatBoost categorical handling and tree models. | UID count adds a pass over grouped rows; already implemented. Reuse existing artifact rather than recomputing for every model. |
| **R2: Full causal transaction features** | Add prior group counts, amount/D9 means/stds, combined card/address/email groups, and amount parts. Winner's notebook reports 262 engineered features and large CPU/GPU preprocessing speed differences. | Global counts/aggregates calculated over full data would reveal future frequency/values. Compute prior-only features; do not derive features from `isFraud`. Avoid high-cardinality static frequency maps for XGBoost. | All three GBDTs. XGBoost receives low-cardinality one-hot columns plus causal numeric counts; LGBM/CatBoost retain native categories. | More groupby/aggregate work and larger artifacts. Generate once; preliminary estimate around the existing 130-second CPU baseline to be measured after approval. GPU acceleration is optional and not required. |
| **R3: R2 + identity/device data** | Winner notebook joins identity data; device/browser/network attributes can add complementary signals. | Left-join on `TransactionID`, verify one row per transaction, retain missingness indicators, and do not use target-derived identity aggregates. Identity coverage is partial, so missing must not be interpreted as a fraud label. | All three GBDTs; CatBoost/LightGBM handle categorical identity fields natively; XGBoost uses low-cardinality one-hot plus numeric encodings. | Adds a **26.5 MB CSV** and extra columns; joins once and saves as a separate artifact. Exact RAM/runtime impact must be measured after approval. |
| **Windowed prior counts** | Top-ten write-up describes transaction counts/types over a time window for UID groups, potentially capturing recent bursts. | Window must end strictly before each transaction timestamp. Window duration is a new hyperparameter and must be selected on validation only. | GBDTs; useful anomaly inputs as numeric features. | Extra rolling/group-window computation and memory; start with a small predeclared set of windows if approved. |
| **D-feature normalization / time features** | Winner solution studies D columns and transaction time; converting relative deltas into day-relative features may expose repeat-transaction behavior. | D-field semantics are masked; test transformations as a separate block, never use future-derived normalization or labels. | GBDTs; scale/standardize only for distance/kernel anomaly methods. | Low-to-medium; vectorized transforms are inexpensive relative to group aggregation. Treat value as unmeasured until validation. |
| **Target encoding** | Common competition technique for high-cardinality categories. | High leakage risk. A full-data target mean leaks each row's label and future labels. Exclude from primary grid; only consider later as strictly chronological out-of-fold/ordered encoding if justified. | CatBoost has native categorical processing; LightGBM/XGBoost may use safely encoded features. | Medium/high implementation and validation burden; not recommended for the initial approved grid. |
| **Identity missingness indicators** | Identity values are not present for all transactions; explicit presence can distinguish “identity observed” from missing fields. | Indicator should encode data availability only, never labels. | All models. | Very low cost; include in identity recipe. |
| **One-class numeric preparation** | Enables a secondary anomaly-detection comparison without training on fraud examples. | Fit transforms only on genuine fit-period rows; select anomaly threshold using validation labels; simulation labels are evaluation-only. | Isolation Forest and scalable one-class SVM. | Numeric scaling/imputation plus categorical encoding. Classic kernel One-Class SVM may be too slow at hundreds of thousands of rows; use a scalable formulation and record runtime. |

## 4. Recommended model and recipe grid

| Family | Include? | Reason |
|---|---|---|
| LightGBM | **Yes** | Existing baseline; native categorical support; winning ensemble family; provides continuity with the bandit result. |
| XGBoost | **Yes** | Used in the first-place solution; strong competition score and complementary tree-boosting behavior. |
| CatBoost | **Yes** | Used in the winning ensemble; handles categorical fields natively; a time-based paper also reports it as a strong PR-AUC model. |
| Prediction blend | **After single-model validation** | Winner and high-ranking solutions blended diverse GBDTs. Build a simple rank/score average only if validation predictions show complementary errors; do not use simulation labels to choose blend weights. |
| Isolation Forest | **Secondary** | Provides an unsupervised anomaly baseline; not a winning competition model. |
| Scalable One-Class SVM | **Secondary** | Tests the one-class question while avoiding the severe runtime of a full kernel SVM on this dataset. |
| Deep learning | **Defer** | Out of scope for this phase per prior project decision. |

Four recipes × three supervised models gives **12 supervised runs**. The two anomaly detectors each get transaction-only and identity-augmented inputs, adding **4 runs**. First screen with seed 42; repeat the strongest valid candidates using seeds 42/43/44. Preprocessing artifact generation is shared across models so each run does not repeat expensive feature engineering.

## 5. Efficiency and expected runtime

| Workload | Evidence / estimate |
|---|---|
| Current preprocessing | Measured **~130 seconds** on Kaggle CPU for 590,540 rows and a 423-column artifact. |
| Winner-inspired CPU feature engineering | Winner notebook reports **~5 minutes per 1 million rows** for its Pandas preprocessing; rough normalization to our data volume is ~3 minutes, but columns, code, hardware, and work differ. Treat only as an order-of-magnitude comparison. |
| Winner RAPIDS GPU feature engineering | Reported **~20 seconds per million rows** versus ~5 minutes CPU in the author's environment. This is not a promise for current Kaggle hardware or our causal version. |
| Model screen | Current LightGBM training stage measured **~30.5 minutes** for the selected workflow including weight comparisons and validation. Exact XGBoost/CatBoost and recipe runtimes are unknown until run. |
| Full matrix | 16 independent model/recipe screening runs; parallel sessions reduce wall time but may compete for CPU/RAM and queue. Kaggle session availability is account/resource dependent. A reliable end-to-end ETA is not available until XGBoost, CatBoost, and anomaly-model timings are measured. The next approval-stage run should capture each notebook's actual runtime and peak-memory diagnostics. |

The SOTA RAPIDS notebook used GPU code and an older GPU/software stack. Our first pass can remain Kaggle CPU-only. The existing project data preparation is already reasonably fast on CPU, so the main cost will likely be the 16 model fits, not repeating preprocessing for each model.

## 6. Recommendation for approval

Approve the following experiment design before implementation:

1. **Preprocessing:** compare R0, R1, R2, and R3; include a small prior-window-count block only if its window definitions are specified before running. Keep target encoding out of the initial grid.
2. **Supervised models:** LightGBM, XGBoost, and CatBoost across all four recipes; test blending after single-model validation.
3. **One-class models:** Isolation Forest and scalable One-Class SVM on transaction-only and identity-augmented inputs, as secondary baselines.
4. **Validation and safety:** retain 48/12/40 chronology, use all fit rows, calculate histories using strictly earlier timestamps, fit encoders/calibrators on allowed development data only, and keep the final 40% out of feature/model selection. Because it has already been examined, describe its new results as exploratory.
5. **After model/preprocessing approval:** every model+recipe that passes validation receives all three identical bandit branches: blocked labels hidden, blocked labels released after **14 days**, and immediate-feedback oracle. Keep costs, action order, feedback delay/censor assumptions apart from this branch, and the 3% review cap unchanged.

## Sources

- [NVIDIA, “Leveraging Machine Learning to Detect Fraud”](https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-develop-a-winning-kaggle-solution/) — first-place model families, feature engineering, validation, leaderboard metrics, and reported preprocessing runtime.
- [Kaggle first-place RAPIDS feature-engineering notebook](https://www.kaggle.com/code/cdeotte/rapids-feature-engineering-fraud-0-96) — source notebook, implementation details, generated features, and speed claim.
- [Kaggle feature-engineering discussion](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/108575) — practical categorical, combination, frequency, and aggregation feature ideas.
- [Kaggle 10th-place solution write-up](https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/m5-10th-solution-and-code) — UID/time-window count features and rare-UID overfit controls.
- [Recent chronological IEEE-CIS study](https://www.mdpi.com/2076-3417/16/12/5809) — time-based PR-AUC model-family reference.
- [Kaggle competition leaderboard](https://www.kaggle.com/competitions/ieee-fraud-detection/leaderboard) — final public/private leaderboard context; its evaluation metric is ROC-AUC.
