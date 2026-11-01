# IEEE-CIS Fraud Detection — Run Log

Concise record of completed stages and their settings. Times are Kaggle runtime, rounded; they exclude setup and queue time. Validation metrics guide model choices. Final research comparisons use the held-out simulation period.

## Project run settings

| Setting | Value |
|---|---|
| Compute | Kaggle CPU notebook; no CUDA/GPU |
| Data | IEEE-CIS Fraud Detection, transaction table; identity remains separate |
| Chronological split | First 60% development; last 20% of development for validation; final 40% simulation |
| Modeling rule | All fit rows retained; compare class weights; no SMOTE/downsampling |
| Bandit protocol | Fixed score model; bandit alone updates on released feedback; 3% review cap |

## Stage 1 — Preprocessing

| Item | Record |
|---|---|
| Notebook | [Kaggle preprocessing](https://www.kaggle.com/code/xerxes187/triple-a-ieee-cis-01-preprocessing) |
| Status / runtime | Completed; about 130 seconds |
| Work performed | Inspected the transaction data and created chronological, leakage-safe feature artifact. |
| Output | 590,540 rows × 423 columns; 20 categorical columns. Feature blocks include 4 frequency, 1 UID, 13 aggregate, and 2 amount-part features. |
| Caveat | UID and other history features are prior-only; identity data is not included in this experiment. |

## Stage 2 — LightGBM training and feature selection

| Item | Record |
|---|---|
| Notebook / run | [Kaggle model training](https://www.kaggle.com/code/xerxes187/triple-a-ieee-cis-02-model-training) · version 4 · run 364460712 |
| Status / runtime | Completed; about 30 minutes 30 seconds |
| Work performed | LightGBM feature ablation; compared `none`, `sqrt`, and `balanced` positive-class weighting for candidate recipes; validation-only isotonic calibration and threshold policy tuning. |
| Selected recipe | `uid` feature block; class-weight strategy `none` |
| Validation PR-AUC | 0.5310 |
| Validation ROC-AUC | 0.8792 |
| Validation precision / recall / F1 at 0.5 | 0.8735 / 0.3511 / 0.5009 |
| Validation Brier score | 0.02145 |
| Validation tuned decision cost | 245,028.6 cost points |
| Interpretation | These are validation results used for selection; the validation cost is not a final held-out result. The historical Phase 1 cost of 720,345 is a reference only. |
| Progress display | This run predates the progress-bar update. Local training code now shows a `tqdm` percentage bar with completed fits and current model/recipe/class weight; publish that code to Kaggle before the next training run. |

## Stage 3 — LightGBM contextual bandit simulation

| Item | Record |
|---|---|
| Notebook | [Kaggle bandit simulation](https://www.kaggle.com/code/xerxes187/triple-a-ieee-cis-03-bandit-simulation) |
| Run / status / runtime | Version 2 · run 364468815 · completed successfully in about 96 seconds of Kaggle CPU runtime. |
| Inputs | Preprocessing output version 8 and LightGBM training output version 4; same final 40% transaction stream for both policies. |
| Simulation setup | LightGBM scores held fixed; bandit alone updates. Seed 42. Feedback: review +1 day; genuine approvals +30 days; fraud approvals randomized from 1–30 days with 20% censored. Blocked labels never released. Hard per-day review cap: 3%. |
| PR-AUC / ROC-AUC on simulation labels | 0.4854 / 0.8674 (diagnostic evaluation only; labels are not provided to the learner). |
| Precision / recall / F1 at 0.5 | 0.8064 / 0.3336 / 0.4720. Brier score 0.02420; ECE 0.00391. |

### Simulation costs and actions

| Experiment | Decision cost | Reviews | Review rate | Cap use | Approve | Block |
|---|---:|---:|---:|---:|---:|---:|
| Closed-loop bandit | 2,870,297.06 | 7,047 | 2.983% | 100% | 25,389 | 203,780 |
| Immediate-feedback oracle | 1,390,336.54 | 7,047 | 2.983% | 100% | 218,034 | 11,135 |
| Always approve | 1,406,054.18 | 0 | 0% | 0% | 236,216 | 0 |
| Always block | 3,073,839.65 | 0 | 0% | 0% | 0 | 236,216 |
| No-review cost policy | 970,061.39 | 0 | 0% | 0% | 224,248 | 11,968 |
| Validation-tuned thresholds | 898,096.88 | 7,047 | 2.983% | 100% | 221,306 | 7,863 |

| Derived comparison | Result |
|---|---:|
| Oracle tax (closed-loop minus oracle) | 1,479,960.51 |
| Closed-loop cost change vs always approve | 104.14% higher |
| Closed-loop cost change vs no-review policy | 195.89% higher |
| Closed-loop cost change vs validation-tuned thresholds | 219.60% higher |
| Closed-loop cost change vs historical Phase 1 reference (720,345) | 298.46% higher; reference is not directly comparable because Phase 1 tuned/evaluated thresholds on the same held-out period |

| Protocol check | Observation |
|---|---|
| Review cap | Respected: 7,047 reviews, exactly the total daily capacity available; no overspend shown in output. |
| Feedback censoring | 203,780 blocked labels remained hidden from the closed-loop learner; 204 approved fraud outcomes were censored; 3,163 feedback events were pending at the end. |
| Result | **Bandit validation failed on decision cost.** Although the simulator completed and enforced the cap, closed-loop policy cost is substantially worse than no-review and validation-tuned baselines. The closed-loop policy blocked most transactions. Do not treat this as a bandit benefit or proceed to other model families until the policy behavior and learning update are diagnosed. |
| Diagnostic clue | Only 29,069 of 236,216 closed-loop decisions generated feedback updates, while every one of the 203,780 Block actions was permanently censored. This action-dependent missing feedback can bias learning and may explain the gap, but needs to be separated from any implementation issue before changing the method. |
| Saved output | Kaggle output includes `artifacts/simulation/metrics.csv`, per-mode decision CSVs/metrics JSON, and cost/detection plots. A concise local copy of the metrics is at `artifacts/simulation/metrics.csv`. |
| Next action | Inspect reward scaling, delayed policy-gradient updates, and action selection; fix and re-run LightGBM bandit under unchanged cost, delay, censoring, seed, and cap settings. |

## Open methodology decision — labels for blocked transactions

| Question | Current recommendation |
|---|---|
| Should the primary run reveal blocked labels? | Keep them hidden for the primary run if the study represents a system that receives no outcome feedback after declining a transaction. IEEE-CIS supplies offline fraud labels for evaluation, but that does not establish that a real system would learn those labels after a block. |
| What does permanent censoring mean? | The learner cannot observe the Block outcome, so it cannot directly learn the false-decline cost of blocking. This is a genuine information limitation, not something inverse-probability weighting can fix when the probability of observing a blocked label is zero. The current result (86% blocked in closed loop) makes this limitation visible. |
| Recommended comparison | Keep the current `blocked labels never released` run as the primary assumption; add a separately named sensitivity run where blocked labels arrive after a declared delay (for example, 30 days), with the other feedback rules unchanged. Keep the existing immediate-after-action oracle as the upper-information comparison. Treat the 30-day blocked-label delay as an assumption unless an external source justifies it. |
| Method reference | Algorithms for delayed and censored bandit feedback explicitly account for observations that have not arrived; this is a different learning problem from ordinary immediate-feedback bandits. See [Vernade et al., Linear Bandits with Stochastic Delayed Feedback](https://proceedings.mlr.press/v119/vernade20a.html). |
| Status | The historical run is preserved at Kaggle notebook version 2. The seed-42 preflight is being rerun with the same blocked-label-hidden default before the new sensitivity branches launch. |

## Stage 4 — Feedback timing sensitivity (exploratory)

The final 40% transaction period has already been evaluated in Stage 3. These runs are therefore sensitivity experiments about feedback assumptions, not independent confirmation and not a basis for choosing a winning policy.

| Branch | Block-label rule | Run plan |
|---|---|---|
| Blocked labels hidden | Never released to closed-loop learner | Seeds 42, 43, 44; closed-loop and matched immediate-feedback oracle |
| Blocked labels after 30 days | Every blocked outcome released exactly 30 days after its decision; no additional censoring | Seeds 42, 43, 44; closed-loop and matched immediate-feedback oracle |
| Immediate-feedback oracle | Outcome released only after its action | Seeds 42, 43, 44; oracle mode only |

| Fixed across branches | Value |
|---|---|
| Score model and features | Saved calibrated LightGBM scores; selected `uid` feature recipe |
| Learner / transaction stream | Existing contextual bandit, held-out rows, transaction order, and settings |
| Other feedback rules | Reviews +1 day; approved genuine +30 days; approved fraud random 1–30 days with 20% censored |
| Costs / review cap | Existing cost table; hard 3% per-day cap |
| Compute | Separate private Kaggle CPU notebook runs; no GPU |
| Outputs | Branch-specific per-seed decisions and metrics, combined per-branch CSV, seed mean/std CSV, and cost plot |

| Local implementation | Record |
|---|---|
| Block-label configuration | `FeedbackConfig.blocked_label_delay_days`; `None` preserves the historical hidden-label behavior |
| Feedback diagnostics | Separate scheduled, released-by-horizon, pending, and permanently hidden Block counts; feedback updates counted by action |
| Preflight | Kaggle simulation notebook version 10, **Seed 42 Preflight Fixed Bundle**; completed successfully in about 102 seconds. It reproduced the historical seed-42 action counts and costs within 0.01 point and passed daily review-cap checks. |
| Preflight result | Closed-loop cost 2,870,297.0555; oracle cost 1,390,336.5431; oracle tax 1,479,960.5124. Closed-loop actions: approve 25,389, review 7,047, block 203,780. Oracle actions: approve 218,034, review 7,047, block 11,135. |
| Feedback checks | Hidden branch kept all 203,780 Block outcomes unavailable to the closed-loop learner. Closed-loop feedback updates 29,069; 3,163 events remain pending at the horizon. Immediate oracle updated on all 236,216 outcomes after each action. |
| Notebook files | `notebooks/03p_bandit_preflight.ipynb` and `notebooks/triple-a-ieee-cis-03[abc]-*.ipynb` |
| Status | Seed-42 preflight passed on Kaggle version 10. All three corrected sensitivity branches subsequently completed; full results are recorded below. |
| Implementation note | Kaggle MCP `save_notebook` does not retain attached notebook outputs, so the inputs were attached in the Kaggle editor for the preflight. For new branch notebooks, inputs must likewise be attached through the editor before each run. The historical Stage 3 version 2 remains intact. |

### Stage 4 execution log

| Branch | Kaggle version | Current status | Notes |
|---|---:|---|---|
| Blocked labels hidden | 2 | Complete; 227 seconds | Initial attempt failed because generated notebook code referenced an undefined delay constant. Corrected to `None` and changed feedback RNG to use the loop seed. |
| Blocked labels after 30 days | 3 | Complete; 241 seconds | Initial attempts failed because generated notebook code referenced undefined delay constants. Corrected to `30` and changed feedback RNG to use the loop seed. |
| Immediate-feedback oracle | 2 | Complete; 145 seconds | Corrected delay constant to `0`, feedback RNG uses the loop seed; oracle still releases labels only after action. |

The failed attempts did not produce usable experiment results; they were caused by invalid generated delay assignments, not simulation outcomes. The corrected runs used separate output folders and both upstream notebook outputs as inputs. All three corrected versions reached Kaggle status `COMPLETE`.

### Stage 4 results — mean ± standard deviation across seeds

| Feedback condition | Closed-loop cost | Oracle cost | Oracle tax | Closed-loop actions (approve / review / block) | Reviews / daily cap | Feedback updates |
|---|---:|---:|---:|---:|---:|---:|
| Blocked labels hidden | 2,883,768.62 ± 12,939.50 | 1,382,234.68 ± 8,299.71 | 1,501,533.93 ± 18,719.29 | 23,749.67 / 7,047 / 205,419.33 | 7,047 / 7,047 (100%) | 27,504.67 ± 1,572.06 closed loop; 236,216 oracle |
| Blocked labels released after 30 days | 1,876,874.63 ± 11,928.92 | 1,382,234.68 ± 8,299.71 | 494,639.94 ± 17,762.71 | 150,263.67 / 7,047 / 78,905.33 | 7,047 / 7,047 (100%) | 153,805.00 ± 11.36 closed loop; 236,216 oracle |
| Immediate-feedback oracle | — | 1,382,234.68 ± 8,299.71 | 0 | 218,412.33 / 7,047 / 10,756.67 (oracle actions) | 7,047 / 7,047 (100%) | 236,216 |

| Diagnostic / baseline | Result |
|---|---:|
| Simulation PR-AUC (same fixed LightGBM scores in every branch) | 0.485435 |
| Precision / recall at probability threshold 0.5 | 0.806353 / 0.333602 |
| No-review baseline cost | 970,061.39 |
| Validation-tuned threshold baseline cost | 898,096.88 |
| Always-approve / always-block costs | 1,406,054.18 / 3,073,839.65 |
| Mean closed-loop cost reduction from 30-day blocked feedback vs hidden feedback | 1,006,893.99 points (34.9%) |

**Interpretation:** Delayed labels for blocked transactions changed learning materially and reduced closed-loop cost by about 34.9% compared with permanently hiding them. The resulting mean cost (1.877M) is still about 93.5% above the no-review baseline and 109.0% above the validation-tuned threshold baseline. The immediate-feedback oracle is also more expensive than these baselines. This sensitivity experiment therefore does not establish a benefit from the current bandit policy. The model PR-AUC/precision/recall figures are fixed-score diagnostics, not measurements of policy-action precision or recall. All branches used the same 3% daily cap and filled the available review capacity; the assertions confirmed no daily cap violation.

**Feedback accounting note:** In oracle runs, the learner received every outcome after its action (236,216 updates), including blocked outcomes, with no pending or censored feedback. The `blocked_labels_released_by_horizon` metric describes the configured evaluator delay schedule; in oracle mode it is not the count of learner updates. Use `feedback_updates` and the per-action update counters to describe actual oracle feedback. In closed loop, the hidden branch released zero blocked outcomes; it averaged 3,099.67 ± 182.44 total feedback events pending at the horizon and 192.33 ± 10.41 approved-fraud labels permanently censored per seed. The 30-day branch scheduled 78,905.33 ± 1,683.14 blocked outcomes per seed on average, released 76,520.67 ± 1,733.55 within the simulation horizon, and ended with 2,384.67 ± 71.67 blocked labels pending. It also averaged 1,074.00 ± 25.36 approved-fraud labels permanently censored. Blocked outcomes were not permanently censored in that condition.

| Local result artifact | Contents |
|---|---|
| [`feedback_sensitivity_metrics.csv`](artifacts/simulation/feedback_sensitivity_metrics.csv) | Per-seed branch metrics plus common baseline rows |
| [`feedback_sensitivity_summary.csv`](artifacts/simulation/feedback_sensitivity_summary.csv) | Mean and sample standard deviation for each branch and mode |
| [`feedback_sensitivity_costs.svg`](artifacts/simulation/feedback_sensitivity_costs.svg) | Cost comparison with no-review and validation-tuned threshold references |

Kaggle outputs remain available in [blocked labels hidden](https://www.kaggle.com/code/xerxes187/triple-a-ieee-cis-03a-blocked-labels-hidden), [blocked labels after 30 days](https://www.kaggle.com/code/xerxes187/triple-a-ieee-cis-03b-blocked-labels-after-30-days), and [immediate-feedback oracle](https://www.kaggle.com/code/xerxes187/triple-a-ieee-cis-03c-immediate-feedback-oracle). This is exploratory reuse of an already analyzed holdout period, not independent evaluation or a basis for selecting a final policy.

## Stage 5 — SOTA-informed expansion (approved plan)

| Item | Status / record |
|---|---|
| Approval gate | Research/preprocessing report delivered in `SOTA_PREPROCESSING_REVIEW.md`; user approved proceeding with its recommendations and the 14-day blocked-label branch. |
| Local implementation | Added independent recipe/model training entry point; added action correctness counts; added Isolation Forest and SGD One-Class SVM secondary pipeline; generated 12 supervised training notebooks, 4 anomaly training notebooks, and 16 matching simulation notebooks. All 40 local notebook files parse as valid notebook JSON. |
| Approved supervised grid | LightGBM, XGBoost, CatBoost × R0 control, R1 UID, R2 full causal transaction, R3 R2 + identity. All use seed 42 initial screen and compare `none`, `sqrt`, `balanced` weights. |
| Feedback protocol in simulation templates | Hidden Block labels; Block labels released after 14 days; immediate-feedback oracle after the action. All retain the existing review/approved feedback rules, cost table, daily 3% review cap, and transaction order. |
| Local files | `src/fraudlab/pipeline.py`, `src/fraudlab/anomaly.py`, `src/fraudlab/metrics.py`; generated `notebooks/grid_*.ipynb` and `notebooks/sim_*.ipynb`. |
| Synthetic smoke check | Kaggle notebook [Triple A Pipeline Smoke Checks — Synthetic Only](https://www.kaggle.com/code/xerxes187/triple-a-pipeline-smoke-checks-synthetic-only), version 1, completed. All 13 Python modules compiled; strict timestamp ties, target-independent features, one-to-one identity join/order, per-action correctness counts, daily 3% cap, hidden/delayed labels, and post-action oracle updates passed on synthetic data. |
| Cloud status | Kaggle account access is active. Preprocessing versions 9–11 failed before reading CSVs because MCP saves dropped the requested competition source. The user attached the competition through Kaggle's Input panel; version 12 then completed successfully. |
| Constraint confirmed | Kaggle's own MCP documentation lists the source fields, but the connected save tool silently drops them in this environment. Kaggle’s product-feedback report documents the same `save_notebook` attachment defect. |
| Preprocessing run | Kaggle [preprocessing notebook](https://www.kaggle.com/code/xerxes187/triple-a-ieee-cis-01-preprocessing), version 12; completed in about 2.0 minutes. Input source `ieee-fraud-detection` is attached. |
| Preprocessing results | 590,540 transaction rows × 394 source columns; transaction feature artifact 590,540 × 423; identity-joined artifact 590,540 × 464. Transaction feature profiles, categorical top values, weekly fraud rates, and identity profile were saved. |
| Run status | **Stage 5 preprocessing completed.** Model/recipe and bandit comparisons have not run yet; no new model metrics, SOTA gap, or bandit conclusion is available. Existing Stage 1–4 results above remain historical and unchanged. |
| Remaining cloud constraint | Kaggle MCP still drops notebook/competition input attachments when saving notebooks. The competition input was attached through the Kaggle UI. Downstream jobs must likewise attach the preprocessing output and model outputs through Kaggle's Input panel, or use a local Kaggle API/CLI credential (none is configured in the local environment). |

Kaggle reference for this blocker: [official MCP documentation](https://www.kaggle.com/docs/mcp) and [Kaggle product-feedback issue reporting dropped data-source fields](https://www.kaggle.com/discussions/product-feedback/727647). The saved preprocessing versions are retained for audit; their failed runs are not data evidence.


| LightGBM R0 control run | Kaggle version 4; preprocessing output v12 attached; seed 42; **COMPLETE** in 419.7 seconds (about 7.0 minutes). Validation PR-AUC 0.529984; ROC-AUC 0.876184; precision 0.861377, recall 0.363600 and F1 0.511351 at p=0.5; Brier score 0.021439; ECE reported 3.9e-18 (needs calibration-metric audit); validation decision cost 245,307.18. Model, metadata, thresholds, validation predictions, and final-40% simulation predictions were saved. This is model scoring only: the requested bandit branches have **not** run for this candidate yet. |
