# IEEE-CIS Fraud Detection: fixed score models + delayed-feedback bandit

This project implements the research pipeline described in the Triple A Phase 1 report and slides. It starts from a new codebase because no implementation repository or dataset was present in the supplied workspace.

## Research protocol

- Dataset: Kaggle IEEE-CIS `train_transaction.csv`; `train_identity.csv` is a separate R3 identity experiment. The hidden-label competition test file is not used.
- Rows are stably sorted by `TransactionDT` and `TransactionID`.
- First 60% of rows is the training period. Its first 80% (48% of all rows) fits each score model; its final 20% (12% of all rows) is validation for class-weight selection, isotonic calibration, and threshold tuning. The final 40% is the simulation stream.
- The score model is fixed during each simulation. Only the contextual bandit updates from feedback the simulator has released.
- The bandit context contains calibrated fraud probability, transaction amount, and remaining daily review capacity. Legal actions are approve, review, and block.
- Review capacity is `floor(3% × rows in that daily batch)`. Once exhausted, Review is masked from the policy.
- Cost points: approving fraud costs the amount; approving genuine costs 0; blocking fraud costs 0; blocking genuine costs 10% of amount; every review costs 5 points.
- Closed-loop base feedback: review labels arrive after 1 day; approved genuine labels arrive after 30 quiet days; approved fraud labels arrive after a random 1–30 days or are censored with probability 20%; the hidden-Block branch never releases Block outcomes. The added sensitivity branch releases them after 14 days.
- The timing values are explicit starting assumptions because the source data contains no review or chargeback records. Sensitivity runs vary maturity/delay and censoring. Hidden outcomes are used for evaluation cost only, never supplied to the closed-loop policy for blocked or pending transactions.
- Oracle feedback reveals each result immediately after its action, including outcomes for blocked transactions. It does not reveal a current transaction's outcome before action selection. Oracle Tax = closed-loop cost − oracle cost.
- All stochastic operations use recorded seeds. Default seed is 42.

## SOTA-inspired model and preprocessing grid

The approved comparison grid consists of four recipes — R0 control, R1 UID, R2 full causal transaction features, and R3 R2 plus one-to-one identity fields — across LightGBM, XGBoost, and CatBoost. Every supervised candidate uses the same 48% fit / 12% validation / 40% simulation split and compares `none`, square-root, and balanced positive-class weights. Identity fields are only included in R3.

Secondary anomaly baselines are Isolation Forest and SGD One-Class SVM on R2 and R3. They fit genuine fit-period rows only and use numeric features from the approved recipe. Their anomaly scores are calibrated on validation only. They are not presumed to match supervised competition performance.

Each model/recipe has a separate notebook under `notebooks/grid_*.ipynb`; matching simulation templates are under `notebooks/sim_*.ipynb`. Simulation uses the same three feedback conditions for every candidate: blocked labels hidden, blocked labels delayed 14 days, and immediate feedback after each chosen action (oracle). Seed 42 is the initial screen; stronger candidates can be repeated with seeds 42, 43, and 44.

## Project layout

```text
data/raw/          Put downloaded Kaggle CSV files here; never commit raw data.
notebooks/         01_preprocessing.ipynb, 02_model_training.ipynb, 03_bandit_simulation.ipynb
src/fraudlab/      Data, features, models, policy, bandit, simulator, metrics, plots, pipeline
artifacts/         Generated data profiles, fitted models, predictions, metrics, and plots
tests/             Small invariant checks for split, costs, capacity, and feedback visibility
```

The identity file is joined only for the R3 identity experiment; R0–R2 remain transaction-only.

## Feature engineering and source

The preprocessing is a **causal adaptation** of public IEEE-CIS competition techniques, rather than a copy of a leaderboard notebook. The first-place solution and competition discussions describe useful patterns such as frequency features, combined card/address/email identifiers, a card/address/D1-day UID, and group aggregates: [first-place notebook](https://www.kaggle.com/code/cdeotte/rapids-feature-engineering-fraud-0-96) and [feature-engineering discussion](https://www.kaggle.com/competitions/ieee-fraud-detection/discussion/108575). Our research replay has a chronological stream, so each history count/mean/std is computed using only strictly earlier `TransactionDT` values; same-time transactions are processed as one batch. The feature code is original project code implementing those general ideas.

Notebook 01 saves the existing simple features as the **control** and stores candidate columns in separate blocks:

- `frequency`: causal prior counts for card2, card3, address, and card/address/email groups.
- `uid`: prior count for card1 + addr1 + transaction day inferred from D1.
- `aggregates`: prior amount and D9 means/stds for card, address, email, and UID groups.
- `amount_parts`: dollar and cents components.

The recipe grid compares each approved recipe independently. The final 40% simulation period is not used for model, weight, or recipe selection. LightGBM/CatBoost retain native categorical handling. XGBoost one-hot encodes categories with at most 20 values and drops high-cardinality raw category IDs; it uses causal count features instead of static frequency maps.

## Local setup

Python 3.10 or newer is required. From the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:IEEE_CIS_DATA_DIR = 'C:\path\to\folder\with\train_transaction.csv'
$env:IEEE_CIS_OUTPUT_DIR = '.\artifacts'
jupyter lab
```

The data loader also searches `data/raw/` and common Kaggle competition mounts. Set `IEEE_CIS_DATA_DIR` if the competition slug differs or you use a local folder.

## Kaggle CPU workflow (recommended first)

1. Accept the IEEE-CIS competition rules in your Kaggle account and make the competition data available to a notebook.
2. Upload this project's `src/` folder and `requirements.txt` as a private Kaggle Dataset named `ieee-cis-fraud-code` (or set `FRAUDLAB_SRC` to its mounted `src` directory). Upload the three notebook files as Kaggle notebooks.
3. Attach the competition dataset and the code dataset as notebook inputs. Enable Internet only if Kaggle needs to install packages; no GPU/CUDA accelerator is required.
4. Install project dependencies in the notebook session with `!pip install -r /kaggle/input/ieee-cis-fraud-code/requirements.txt` (the code dataset should contain both `src/` and `requirements.txt`).
5. Run `01_preprocessing.ipynb`. It writes profile CSV/JSON and `features/ieee_cis_features.parquet` beneath `/kaggle/working/artifacts`.
6. Run `02_model_training.ipynb` with `MODEL_NAMES = ['lightgbm']`.
7. Run `03_bandit_simulation.ipynb` with `MODEL_NAMES = ['lightgbm']`. Review the LightGBM closed-loop, oracle, and threshold-baseline metrics before expanding.
8. After that gate passes, run `02` with XGBoost and CatBoost, then `03` with the same model names and the same artifacts/configuration.
9. For the approved expansion, run the matching independent `grid_<model>_<recipe>.ipynb` notebooks, then the matching `sim_<model>_<recipe>.ipynb` notebooks. Every included candidate must use all three feedback branches. Use seed 42 for the full screen; repeat validation-selected finalists with seeds 42, 43, and 44.
10. Download `/kaggle/working/outputs` from notebook outputs so results survive session cleanup. For a long unattended run, use Kaggle's **Save & Run All** rather than leaving an interactive session idle. Kaggle runtime availability and limits can change; check the session's current quota.

Because each Kaggle notebook runs in its own session, pass outputs between stages by saving each prior notebook's output as a Kaggle Dataset (or downloading and re-uploading its artifact folder) and attaching it as an input to the next notebook. Attach both the feature artifact from Notebook 01 and the model outputs from Notebook 02 to Notebook 03. The notebooks search mounted inputs by expected relative file names. The saved Parquet features avoid recomputing causal history features.

Kaggle mounts competition data read-only under `/kaggle/input`. Generated output belongs under `/kaggle/working`. Set `IEEE_CIS_DATA_DIR` for a different competition-data mount, `IEEE_CIS_ARTIFACT_DIR` for the input artifact dataset, `IEEE_CIS_OUTPUT_DIR` for a writable output folder, and `FRAUDLAB_SRC` for the code mount.

## Execution order and gates

1. **Preprocessing:** profile the full training CSV, create causal time/frequency/history features, save Parquet, and verify split chronology and fraud rates.
2. **LightGBM:** compare the control and candidate feature blocks. For each recipe, compare three class-weight strategies (`none`, square-root imbalance, and balanced), select the weight by validation PR-AUC, then fit isotonic calibration on validation predictions only. Choose feature blocks using validation-tuned decision cost, with PR-AUC as a tie-break. Save validation and simulation probabilities.
3. **LightGBM + bandit:** run closed-loop and immediate-label oracle on identical held-out transactions. Confirm action probabilities, no early feedback, blocked-label censoring, daily cap, costs, and reproducible seed behavior. Do not launch other models until this stage is reviewed.
4. **Other score providers:** train XGBoost and CatBoost with the same split, validation protocol, and class-weight candidates. Each gets its own calibrated score file but uses the same bandit implementation and simulator settings.
5. **Sensitivity:** vary feedback maturity/delay and fraud censoring, keeping the baseline configuration and random seeds recorded.

The historical Phase 1 cost of 720,345 points is stored as a reported reference. It is not recomputed as an unbiased holdout result because the report disclosed threshold tuning and evaluation on the same slice.

## Outputs

The pipeline writes dataset and feature profiles, split summaries, weight-search tables, calibrated validation predictions, simulation predictions, model artifacts, decision logs, per-run metrics JSON/CSV, experiment configuration JSON, and cost/detection plots. Raw outcomes exist in the offline artifact for evaluation, but the bandit API receives context/action/reward only when the simulator releases feedback.

Primary outputs include:

- Total decision cost and Oracle Tax.
- Review rate and cap utilization.
- PR-AUC, ROC-AUC, precision, recall, F1, Brier score, and 10-bin expected calibration error.
- Comparison against always-approve, always-block, expected-cost no-review, and validation-tuned threshold policies.

## Reproducibility and runtime

Default settings are in `src/fraudlab/config.py`. Every run saves `experiment_config.json`; model and bandit seeds default to 42. Dependencies are listed in `requirements.txt`. CPU runtimes are estimates until the first full Kaggle run. Fit times depend on installed library versions and current Kaggle allocation; this first version is not tuned for multi-provider parallel execution.

## Known limitations

- Delay, maturity, censoring, and review accuracy are simulator assumptions, not IEEE-CIS measurements.
- Permanently censored blocked outcomes cause action-dependent learning bias by design. The learner cannot directly correct block-related mistakes from those missing labels.
- Oracle Tax is an upper-bound comparison because the oracle receives labels for blocked transactions that the closed-loop learner never sees.
- This first implementation freezes each fraud-score model during the held-out replay. It tests policy adaptation, not daily supervised-model retraining.
- Identity features, neural models, DQN, and multi-provider parallel runs are deferred.

