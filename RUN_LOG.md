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

## Stage 5 execution update — 2026-11-02

| Candidate | Version | Status | Input | Evidence / action |
|---|---:|---|---|---|
| XGBoost R0 Control | 6 | Kaggle API: `RUNNING` | Preprocessing output v12 attached in notebook metadata | Submission via editor Save & Run All; output artifacts/runtime not yet verified. |
| XGBoost R1 UID | 3 | Kaggle API: `RUNNING` | Preprocessing output v12 attached in notebook metadata | Submission via editor Save & Run All; output artifacts/runtime not yet verified. |
| XGBoost R2 Full Causal | 2 draft attempt | Rejected before run | Preprocessing input visible in editor | Kaggle rejected Save & Run All: maximum batch CPU session count of 5 reached. No result produced. |

| Preflight discrepancy | Observation |
|---|---|
| Kaggle notebook status API | XGBoost R0 v6 and R1 v3 both return `RUNNING`. |
| Kaggle Active Events panel | Displays `0 Active Events`. |
| CPU session capacity | Save & Run All rejected R2 with `Maximum batch CPU session count of 5 reached`. |
| Interpretation | Conflicting Kaggle status surfaces; neither queued nor completed is inferred. Runs remain unverified until logs/artifacts confirm execution. No unidentified session was cancelled. |
| Recovery attempted | Reopened R2 in editor, confirmed code and preprocessing attachment, attempted Save & Run All, observed cap rejection, and opened Kaggle's Manage active sessions control; it returned to the editor without exposing a session list. |
| Next | Resolve session accounting through candidate-specific run/version pages and retry on an identified free slot. Do not count legacy `COMPLETE` versions without matching artifacts and configuration. |

### Stage 5 execution correction — 2026-11-02

This entry supersedes the provisional status immediately above. Kaggle run status and notebook version data were rechecked; the API can say `COMPLETE` for stale versions, so completion requires both the current version and expected outputs/logs.

| Candidate | Current version | Confirmed state | Evidence |
|---|---:|---|---|
| XGBoost R0 Control | 6 | Completed; 704.9 seconds | Kaggle log and expected output artifacts. Validation PR-AUC 0.5145723; ROC-AUC 0.8849459; precision 0.8335; recall 0.3333; F1 0.4762 at p=0.5; Brier 0.02223; validation cost 205,491.02. ECE near zero is suspicious and needs audit. Pandas fragmentation warnings appeared. |
| XGBoost R1 UID | 3 | Completed; 772.7 seconds | Kaggle run log says successful. Output contains model, validation/simulation predictions, weight search, threshold file, split summary, config, and run record. Metrics still need extraction and acceptance checks. |
| XGBoost R2 Full Causal | 2 | Running | Kaggle API reports RUNNING; metadata confirms preprocessing v12 input; output not yet present. |
| LightGBM R1 UID | 2 | Not launched/confirmed | Current version remains 2; history only shows quick versions and no training artifacts. An attempted Save & Run All did not create a confirmed run version. |
| CatBoost R0 Control | 2 | Not launched/confirmed | Current version remains 2; no training output artifacts. An attempted Save & Run All did not create a confirmed run version. |

Kaggle's current session-cap behavior is inconsistent across UI/API surfaces. The over-cap Save & Run All attempt was rejected rather than queued. No unidentified session was cancelled. Continue to verify per-notebook versions/logs and submit the next prepared candidate after a slot is genuinely available. Existing `RUNNING` or `COMPLETE` API values alone are not sufficient evidence of a new run.

### Kaggle execution status correction — 2026-11-02 (continued)

| Candidate | Current version | State | Evidence and metrics |
|---|---:|---|---|
| XGBoost R1 UID | 3 | Completed; 772.7 seconds Kaggle elapsed (746.9 seconds measured in the training code) | `run_record.json` retrieved from the completed version. Validation PR-AUC 0.5184881; ROC-AUC 0.8874344; precision 0.80435, recall 0.34342, F1 0.48133 at p=0.5; Brier 0.0222415; validation cost 194,030.24. Compared with XGBoost R0, UID is +0.00392 PR-AUC, +0.00249 ROC-AUC, and validation cost lower by 11,460.78. Calibration ECE reports approximately 3.15e-18, which is implausibly close to exact zero and must be audited before acceptance. |
| XGBoost R0 Control | 6 | Completed; 704.9 seconds | Existing verified metrics retained above. Its near-zero ECE also needs audit. |
| XGBoost R2 Full Causal | 2 | Running | Current session status API reports RUNNING; preprocessing output v12 is attached. |
| CatBoost R0 Control | 3 | Running | Save & Run All was accepted as version 3; current session status API reports RUNNING; preprocessing output v12 is attached. |
| LightGBM R1 UID | 2 | Not launched; Kaggle rejected the submission | Editor showed the CPU batch-session cap rejection (“Maximum batch CPU session count of 5 reached”). The version did not advance; do not count it as queued or running. Preprocessing output v12 is attached. |

Kaggle CPU capacity recovery: the session cap is still reached despite the Active Events panel showing zero events. The panel did not expose a verifiable session list, so no sessions were stopped or cancelled. Wait for the two identified project jobs (XGBoost R2 and CatBoost R0) to finish, then retry LightGBM R1. After one successful finish, refill available capacity with prepared candidates, checking each saved version and input lineage before launch. The fixed cap and contradictory status surfaces are reported as observed; no unverified status is treated as completion.

| XGBoost R1 UID weight-search note | In the saved run record, `none` produced validation PR-AUC 0.528243, `sqrt` 0.518452, and `balanced` 0.503015; the selected calibrated result recorded in the same run was PR-AUC 0.5184881. This apparent gap means the reported headline uses a distinct post-selection/calibration path; audit the training module and run record fields before interpreting comparisons. |
| Calibration audit | ECE around 1e-18 on both XGBoost control and UID is an in-sample calibration score because isotonic calibration was fitted and assessed on the same validation labels. Treat it as optimistic and unsuitable as an independent calibration estimate; audit proper held-out calibration or temporal calibration split before acceptance. |
| CatBoost R0 Control | Version 3 is now accepted by Kaggle and API session status reports RUNNING; preprocessing v12 is attached. A previous attempt hit Kaggle's batch CPU cap, then a later Save & Run All submission succeeded. |
| LightGBM R1 UID | Still not launched. Save & Run All was rejected by the explicit Kaggle five-session batch CPU cap; API status COMPLETE is stale (version2 / no matching new artifacts), not evidence of the requested job. |
| XGBoost R2 Full Causal | Version 2 remains RUNNING; preprocessing v12 is attached. |
| Capacity interpretation | Kaggle's explicit dialog requires stopping one of the other CPU sessions; the Active Events panel reports zero and reveals no identifiable session list. No session was stopped. Wait for the two known project runs to finish, then retry LightGBM R1 and fill slots in controlled waves. |
| XGBoost R2 Full Causal | 2 | Completed; 746.8 seconds code timer | Retrieved run record; output includes model, validation/simulation predictions, thresholds, config, and metadata. Validation PR-AUC 0.5214449; ROC-AUC 0.8839832; precision 0.84474, recall 0.33374, F1 0.47845 at p=0.5; Brier 0.0221625; validation cost 198,372.64. Weight-search PR-AUC: none 0.5297136, sqrt 0.5211845, balanced 0.5003808. ECE 7.97e-19 is an in-sample estimate after isotonic calibration on the same validation data and must not be read as independent calibration performance. |
| LightGBM R1 UID | 3 | RUNNING | Retried immediately after XGBoost R2 completed. The save dialog showed Save & Run All, preprocessing output v12 was attached, and Kaggle session status reports RUNNING. |
| LightGBM R2 Full Causal | 2 | Rejected before run | Notebook configuration and v12 input were verified; Kaggle again displayed the five-session batch CPU cap after Save & Run All. It did not queue or advance to a new run. |
| Current project jobs | — | 3 observed running | XGBoost R1? No: XGBoost R1 is complete. Active identified runs: XGBoost R2 is complete, CatBoost R0 v3 RUNNING, and LightGBM R1 v3 RUNNING. The API has stale `COMPLETE` values for some older versions, so candidate confirmation uses the submitted version/editor event plus session status and outputs. |
| Corrected active-job count | 2 identified model runs are active now: CatBoost R0 Control v3 and LightGBM R1 UID v3. XGBoost R2 Full Causal v2 is complete; XGBoost R1 UID v3 is also complete. This correction supersedes the preceding row that mistakenly said three jobs were running. |
| Progress display | The submitted notebook training cells do not currently emit an explicit percentage progress bar. Keep the active runs unchanged; add a progress bar to future model-training notebook versions before they are submitted. |
| LightGBM R1 UID | 3 | Completed; 396.5 seconds | Retrieved run record and expected artifacts. Validation PR-AUC 0.5310172; ROC-AUC 0.8792340; precision 0.87349, recall 0.35109, F1 0.50086 at p=0.5; Brier 0.0214527; validation cost 245,028.60. Against LightGBM R0, PR-AUC is +0.00103, ROC-AUC +0.00305, and validation cost lower by 278.58. Weight-search PR-AUC: none 0.5409054, sqrt 0.5227474, balanced 0.5177848. ECE 1.73e-19 is in-sample after fitting isotonic calibration on the same validation labels; not an independent calibration result. |

### Latest execution update — 2026-11-02 (continued)

| Candidate | Version | Status | Verified notes |
|---|---:|---|---|
| LightGBM R1 UID | 3 | Complete | Runtime 396.5 s. Validation PR-AUC 0.5310172; ROC-AUC 0.8792340; validation cost 245,028.60. Weight-search PR-AUC: none 0.5409054, sqrt 0.5227474, balanced 0.5177848. Compared with LGBM R0, the selected calibrated headline improves PR-AUC by 0.00103 and cost by 278.58 points. |
| XGBoost R2 Full Causal | 2 | Complete | Training timer 746.8 s. Validation PR-AUC 0.5214449; ROC-AUC 0.8839832; validation cost 198,372.64. Raw weight-search PR-AUC: none 0.5297136, sqrt 0.5211845, balanced 0.5003808. |
| CatBoost R0 Control | 3 | Running at last check | Seed 42, preprocessing output v12 attached. Await candidate-specific logs/artifacts before accepting results. |
| XGBoost R3 Full Causal + Identity | 2 | Running at last check | Seed 42; preprocessing output v12 attached. Await candidate-specific logs/artifacts before accepting results. |
| LightGBM R2 Full Causal | 2 | Rejected by Kaggle CPU cap | Correct model, recipe, seed, and v12 input were verified. Save & Run All showed the explicit five-session batch CPU cap. It did not queue and no new version was created. Retry after an identified slot is freed. |

**Capacity and monitoring correction:** Two candidate jobs were positively identified as running at the last browser check (CatBoost R0 v3 and XGBoost R3 v2). Kaggle's Active Events panel showed zero events despite the batch CPU cap. Do not infer hidden queued work, stop unidentified sessions, or count stale API states as new runs. The cap rejects excess submissions instead of queueing them.

**Progress reporting:** These already-submitted grid notebooks do not show a percentage progress bar during training. Preserve the in-flight notebook versions; add progress bars to any newly prepared/retried training versions before submission so their epoch/iteration progress is visible.

**Calibration caveat:** The near-zero ECE values in these outputs are produced by fitting and scoring isotonic calibration on the same validation labels. They are optimistic and are not independent calibration evidence. The validation results remain useful for screening with that caveat, but proper calibration claims require a held-out calibration window or cross-fitting.

### Verified candidate results and current jobs — 2026-11-02 (latest check)

| Candidate | State | Runtime | Validation PR-AUC | ROC-AUC | Validation cost | Notes |
|---|---|---:|---:|---:|---:|---|
| LightGBM R1 UID v3 | Complete | 396.5 s | 0.5310172 | 0.8792340 | 245,028.60 | Weight search: none 0.5409054, sqrt 0.5227474, balanced 0.5177848. |
| XGBoost R2 Full Causal v2 | Complete | 746.8 s code timer | 0.5214449 | 0.8839832 | 198,372.64 | Weight search: none 0.5297136, sqrt 0.5211845, balanced 0.5003808. |
| CatBoost R0 Control v3 | Running | — | — | — | — | Preprocessing v12 attached; no log/artifact yet. |
| XGBoost R3 Full Causal + Identity v2 | Running | — | — | — | — | Preprocessing v12 attached; no log/artifact yet. |
| LightGBM R2 Full Causal | Rejected | — | — | — | — | Explicit Kaggle five-session batch CPU cap; not queued and no new version. Retry when a known slot becomes available. |

**Current count:** Two identified jobs are running (CatBoost R0 and XGBoost R3); two recently verified candidate runs are complete (LightGBM R1 and XGBoost R2). The LGBM R2 status API's `COMPLETE` label is stale and does not represent a new run. Kaggle output listings for both active jobs still return empty logs.

**Progress indicator requirement:** The submitted training entry cells do not provide percentage progress. Keep the in-flight versions unchanged; add a visible `tqdm`/percentage indicator to any candidate notebook version edited before future submission.

**Calibration metric caveat:** Near-zero ECE is obtained on the same validation labels used for isotonic fitting; it is optimistic. Do not present this as an independent calibration result.

### XGBoost R3 identity result — 2026-11-02

| Candidate | Version | Runtime | Validation PR-AUC | ROC-AUC | Precision / recall / F1 at p=0.5 | Brier | Validation cost |
|---|---:|---:|---:|---:|---|---:|---:|
| XGBoost R3 Full Causal + Identity | 2 | 1,002.5 s | 0.5224685 | 0.8890399 | 0.85333 / 0.33575 / 0.48190 | 0.021956 | 192,371.96 |

Weight-search PR-AUC: none 0.5328924; sqrt 0.5319181; balanced 0.5022712. Best raw PR-AUC is no weighting. The recorded calibrated headline PR-AUC is 0.5224685. The identity recipe reduced validation decision cost relative to XGBoost R2 by 6,000.69 points, while its headline PR-AUC rose by 0.00102 and ROC-AUC by 0.00506. ECE is near zero because isotonic calibration was fitted and assessed on the same validation labels; it is not an independent calibration result. Model plus aligned validation/simulation predictions and metadata are present in version 2 output. This is validation-only screening; do not interpret simulation period results as independent evidence.

| Current cloud status | Candidate-specific evidence |
|---|---|
| XGBoost R3 Full Causal + Identity v2 | Complete; successful run record and expected artifacts retrieved. |
| CatBoost R0 Control v3 | Still reports RUNNING; candidate output/log endpoint remains empty. Kaggle Active Events UI continues to show 0 events, so use the version-history entry and matching artifacts before declaring completion. |
| LightGBM R2 Full Causal | Previous Save & Run All attempts were rejected at Kaggle's five-session CPU limit; no version was queued or completed. Retry after a confirmed capacity slot. |

### LightGBM R1 UID bandit screening run — 2026-11-02

| Item | Status / detail |
|---|---|
| Notebook | `triple-a-bandit-lightgbm-r1-uid`, version 3 |
| Inputs verified in editor | Preprocessing v12 and LightGBM R1 UID model output |
| Configuration | Seed 42; fixed LightGBM scores; 3 feedback branches: blocked hidden, blocked after 14 days, immediate-after-action oracle |
| Progress display | `tqdm` over seeds; the run has one seed in this screen |
| Submission | Kaggle Save & Run All accepted; v3 completed and outputs are available. |

#### LightGBM R1 UID seed-42 bandit results

| Feedback branch | Decision cost | Oracle cost | Oracle Tax | PR-AUC | Precision @ 0.5 | Recall @ 0.5 | Approve | Block | Review | Feedback updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Immediate-after-action oracle | 1,390,336.54 | 1,390,336.54 | 0.00 | 0.485435 | 0.806353 | 0.333602 | 218,034 | 11,135 | 7,047 | 236,216 |
| Blocked labels hidden | 2,870,297.06 | 1,390,336.54 | 1,479,960.51 | 0.485435 | 0.806353 | 0.333602 | 25,389 | 203,780 | 7,047 | 29,069 |
| Blocked labels after 14 days | 2,895,521.55 | 1,390,336.54 | 1,505,185.01 | 0.485435 | 0.806353 | 0.333602 | 21,287 | 207,882 | 7,047 | 197,927 |

The model score stream is fixed, so classification PR-AUC/precision/recall do not change by branch. Feedback availability changes bandit actions and decision costs. All branches used 7,047 reviews under the 3% cap. These are simulation-period results on a period previously examined, so they are exploratory, not independent confirmation.

#### XGBoost R0 Control bandit run

| Item | Status / detail |
|---|---|
| Notebook | `triple-a-bandit-xgboost-r0-control`, version 3 |
| Inputs | Preprocessing v12 and matching XGBoost R0 model output attached and verified |
| Configuration | Seed 42; blocked labels hidden, blocked labels after 14 days, immediate-after-action oracle |
| Submission | Save & Run All accepted; v3 completed in 119 seconds and outputs were verified |
| Results | Retrieved from v3: all three feedback branches; 7,047 reviews each; review cap respected |

#### XGBoost R0 Control seed-42 bandit results

| Feedback branch | Decision cost | Oracle cost | Oracle Tax | PR-AUC | Precision @ 0.5 | Recall @ 0.5 | Approve | Block | Review | Feedback updates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Immediate-after-action oracle | 1,387,446.75 | 1,387,446.75 | 0.00 | 0.492914 | 0.770882 | 0.343631 | 217,850 | 11,319 | 7,047 | 236,216 |
| Blocked labels hidden | 2,872,664.97 | 1,387,446.75 | 1,485,218.21 | 0.492914 | 0.770882 | 0.343631 | 24,898 | 204,271 | 7,047 | 28,627 |
| Blocked labels after 14 days | 2,895,581.86 | 1,387,446.75 | 1,508,135.11 | 0.492914 | 0.770882 | 0.343631 | 21,054 | 208,115 | 7,047 | 197,952 |

The v3 Save & Run All completed in 119 seconds. All branches used 7,047 reviews under the 3% cap. Score classification metrics are invariant across feedback conditions because the score model is fixed; actions/cost differ with feedback. Outputs were retrieved from version label `v3`; version4 is a Quick Save only. These simulation-period results are exploratory because this period was previously examined.

Kaggle concurrency check: CatBoost R0 still reports `RUNNING`; the other queried supervised candidates report `COMPLETE`. Kaggle's API timestamps can lag or differ from the active editor. This run is counted as active only because version 3 was newly created and the session status reports `RUNNING`. No unrelated session was cancelled.

#### XGBoost R1 UID bandit seed-42 screening run — 2026-11-02

| Item | Value |
|---|---|
| Notebook/version | `triple-a-bandit-xgboost-r1-uid`, v2 |
| Inputs | Preprocessing v12 + XGBoost R1 UID prediction output; both visible in the notebook input panel and confirmed in metadata |
| Runtime | Kaggle output artifact confirms all branch CSVs, summaries, JSONs, and plot; status API still says `RUNNING` despite complete outputs, so status is treated as stale |
| Review cap | 7,047 / 236,216 transactions (2.9833%); cap utilization 100% in all branches |
| Fixed score metrics | PR-AUC 0.496087; ROC-AUC 0.879226; precision@0.5 0.751947; recall@0.5 0.356081; F1@0.5 0.483298; Brier 0.0241454; ECE-10 0.0032136 |

| Feedback branch | Decision cost | Oracle cost | Oracle tax | Approve | Block | Review | Feedback updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Immediate-after-action oracle | 1,380,944.97 | 1,380,944.97 | 0.00 | 217,664 | 11,505 | 7,047 | 236,216 |
| Blocked labels hidden | 2,872,167.21 | 1,380,944.97 | 1,491,222.24 | 24,993 | 204,176 | 7,047 | 28,671 |
| Blocked labels after 14 days | 2,895,185.93 | 1,380,944.97 | 1,514,240.96 | 21,067 | 208,102 | 7,047 | 197,950 |

Branch rows have identical classification metrics because the model prediction stream remains fixed. The policy's actions and decision costs vary with label availability. This previously analyzed 40% period is exploratory, not independent confirmation.

### XGBoost R1 UID bandit run completion — 2026-11-02

| Item | Value |
|---|---|
| Notebook/version | `triple-a-bandit-xgboost-r1-uid`, v2 (`v2` output label) |
| Runtime | 144 s from Kaggle output stream |
| Inputs | Preprocessing v12 and matching XGBoost R1 UID training output confirmed in notebook metadata |
| Review cap | 7,047 of 236,216 transactions (2.9833%); cap utilization 100% in all branches |
| Fixed score metrics | PR-AUC 0.496087; ROC-AUC 0.879226; precision@0.5 0.751947; recall@0.5 0.356081; F1@0.5 0.483298; Brier 0.0241454; ECE-10 0.0032136 |

| Feedback branch | Decision cost | Oracle cost | Oracle Tax | Approve | Block | Review | Feedback updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Immediate-after-action oracle | 1,380,944.97 | 1,380,944.97 | 0.00 | 217,664 | 11,505 | 7,047 | 236,216 |
| Blocked labels hidden | 2,872,167.21 | 1,380,944.97 | 1,491,222.24 | 24,993 | 204,176 | 7,047 | 28,671 |
| Blocked labels after 14 days | 2,895,185.93 | 1,380,944.97 | 1,514,240.96 | 21,067 | 208,102 | 7,047 | 197,950 |

All three candidate results retrieved so far are consolidated in `outputs/bandit_screen_metrics.csv`. Simulation costs are exploratory because the chronological simulation period has already been examined.

#### LightGBM R0 control bandit seed-42 completion — 2026-11-02

| Item | Value |
|---|---|
| Notebook/version | `triple-a-bandit-lightgbm-r0-control`, v2 |
| Inputs | Preprocessing v12 + LightGBM R0 model output; both sources verified in Kaggle metadata and editor sidebar |
| Status | Save & Run All completed; all three branch decision files and `metrics.csv` are present |
| Held-out stream | 236,216 transactions; 7,047 reviews in every branch (2.9833%, cap utilization 100%) |
| Fixed score metrics | PR-AUC 0.485809; ROC-AUC 0.866556; precision@0.5 0.787839; recall@0.5 0.345014; F1@0.5 0.479878; Brier 0.024225; ECE-10 0.003713 |

| Feedback branch | Decision cost | Oracle cost | Oracle Tax | Approve | Block | Review | Feedback updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Immediate-after-action oracle | 1,380,844.03 | 1,380,844.03 | 0.00 | 217,765 | 11,404 | 7,047 | 236,216 |
| Blocked labels hidden | 2,862,886.74 | 1,380,844.03 | 1,482,042.71 | 26,364 | 202,805 | 7,047 | 29,801 |
| Blocked labels after 14 days | 2,893,664.29 | 1,380,844.03 | 1,512,820.27 | 21,290 | 207,879 | 7,047 | 197,929 |

The R0 seed-42 bandit rerun passed the three-branch review-cap check. The model score stream stays fixed, so classification metrics are the same for all branches; actions, feedback updates, and costs vary. Results are exploratory on the previously examined simulation period.

### CatBoost R0 bandit sensitivity run — 2026-11-02

| Item | Value |
|---|---|
| Notebook/version | `triple-a-bandit-catboost-r0-control-screen-42`, v2 |
| Inputs | Preprocessing v12 and CatBoost R0 prediction output attached and visible in the notebook input panel |
| Runtime | 147 seconds (Kaggle log) |
| Held-out stream | 236,216 transactions; 7,047 reviews in each branch (2.9833%, 100% of the cap) |
| Model metrics | PR-AUC 0.519628; ROC-AUC 0.893092; precision@0.5 0.768493; recall@0.5 0.388012; F1@0.5 0.515664; Brier 0.023240; ECE-10 0.003267 |

| Feedback branch | Decision cost | Oracle cost | Oracle Tax | Approve | Block | Review | Feedback updates | Feedback pending | Block labels hidden / pending |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Immediate-after-action oracle | 1,336,287.86 | 1,336,287.86 | 0.00 | 216,464 | 12,705 | 7,047 | 236,216 | 0 | 0 / 0 |
| Blocked labels hidden | 2,867,884.41 | 1,336,287.86 | 1,531,596.54 | 25,374 | 203,795 | 7,047 | 29,001 | 3,217 | 203,795 / 0 |
| Blocked labels after 14 days | 2,889,986.35 | 1,336,287.86 | 1,553,698.49 | 21,293 | 207,876 | 7,047 | 197,925 | 38,121 | 0 / 34,961 |

| Feedback branch | Approve genuine correct | Approve fraud missed | Block fraud correct | Block genuine declined | Review fraud | Review genuine |
|---|---:|---:|---:|---:|---:|---:|
| Immediate-after-action oracle | 209,998 | 6,466 | 1,736 | 10,969 | 473 | 6,574 |
| Blocked labels hidden | 24,458 | 916 | 7,516 | 196,279 | 243 | 6,804 |
| Blocked labels after 14 days | 20,514 | 779 | 7,668 | 200,208 | 228 | 6,819 |

All branches pass the hard review-cap check. Hidden-label branch receives no blocked outcomes. The 14-day branch releases 172,915 blocked outcomes by the horizon; 34,961 remain pending, alongside 3,160 non-Block events. Oracle reveals labels only after each action. The fixed score model means classification metrics are invariant across branches. These metrics are exploratory on the previously examined simulation period.

### Parallel training launch correction — 2026-11-02

A bulk attempt to launch four Kaggle training commits hit Kaggle's `Too many requests to CommitAndRun` rate limit. The LightGBM R3 identity job was accepted and is running as version 3. LightGBM R2 and CatBoost R1/R2 were rejected and remain unlaunched. All four notebooks were verified in the editor with preprocessing v12 attached; configs match model, recipe, and seed 42. Continue with one controlled commit at a time after Kaggle permits it, and verify the version-history status before recording a launch.

### LightGBM R3 Full Causal + Identity seed-42 training — 2026-11-02

| Item | Result |
|---|---|
| Kaggle notebook/version | `triple-a-grid-lightgbm-r3-full-causal-identity`, v3 |
| Status/runtime | Complete; 447.9 seconds |
| Input | Preprocessing output v12; transaction and identity feature artifacts |
| Validation PR-AUC | 0.5244251 |
| Validation ROC-AUC | 0.8726397 |
| Precision / recall / F1 at 0.5 | 0.853076 / 0.374899 / 0.520886 |
| Brier / ECE-10 | 0.021506 / approximately 0 (isotonic calibration assessed on its fitting validation window; optimistic) |
| Validation cost | 242,170.05 |
| Outputs | Candidate/config metadata, model, validation and simulation predictions, weight search, thresholds, split summary, run record |

The Kaggle output manifest was present and the candidate is accepted for simulation after verifying its model/recipe/seed and prediction artifacts. Next: run the same three feedback branches for this fixed-score model. Simulation-period costs remain exploratory because this period has already been examined.

### Current remaining seed-42 screen state — 2026-11-02

| Candidate | State |
|---|---|
| LightGBM R3 Full Causal + Identity v3 | Complete; metrics above; bandit simulation pending |
| CatBoost R1 UID v3 | Running; wait for artifacts before accepting |
| LightGBM R2 Full Causal | Notebook v2 only; v3 not submitted |
| CatBoost R2 Full Causal | Notebook v2 only; v3 not submitted |
| CatBoost R3 Full Causal + Identity | Notebook v2 only; v3 not submitted |
| Anomaly candidates | Four planned combinations; execution status to be checked and run after training backlog |

Kaggle had rejected earlier rapid commit attempts. Continue with a paced, controlled submission cadence, verify each version and input lineage, and avoid claiming a queued job unless Kaggle confirms it.

### Kaggle execution audit correction — 2026-11-02

| Candidate | Kaggle status | Execution evidence | Current disposition |
|---|---|---|---|
| LightGBM R3 training, v3 | COMPLETE | Expected run manifest, model, validation/simulation predictions and metric artifacts; 447.9 s recorded | Accepted for bandit simulation |
| LightGBM R3 bandit, v1 | RUNNING | Notebook metadata confirms preprocessing output and matching model output are both attached | Monitor; collect all three branch outputs on completion |
| CatBoost R1 UID, v3 | RUNNING | No validated output manifest yet | Wait before starting its simulation |
| LightGBM R2 causal, v2 | COMPLETE according to status API | Quick Version in history; all cells remain `In [ ]`; output listing contains only notebook conversion log | Not a training run; needs a real Save & Run All |
| CatBoost R2 causal, v2 | COMPLETE according to status API | Quick Version in history; all cells remain `In [ ]`; output listing contains only notebook conversion log | Not a training run; needs a real Save & Run All |

Kaggle rejected recent `CommitAndRun` attempts with `Too many requests`. A status of `COMPLETE` can refer to a quick save with no code execution. Future acceptance requires both executed cells and the expected model artifacts. Continue one validated submission at a time when Kaggle allows it; don't claim rejected jobs are queued.

### LightGBM R3 Full Causal + Identity bandit seed-42 — 2026-11-02

| Item | Result |
|---|---|
| Kaggle notebook/version | `triple-a-bandit-lightgbm-r3-full-causal-identity`, v1 |
| Inputs | Preprocessing output v12 + matching LightGBM R3 training output; both confirmed in notebook metadata |
| Runtime | 136.7 seconds total Kaggle log time; branch metrics printed at 127.7 seconds |
| Stream | 236,216 transactions; same held-out stream for all branches |
| Fixed model score metrics on simulation stream | PR-AUC 0.484229; ROC-AUC 0.866877; precision@0.5 0.776112; recall@0.5 0.352046; F1@0.5 0.484377; Brier 0.024149; ECE-10 0.003169 |
| Review-cap check | 7,047 reviews per branch; 2.9833% overall; 100% of permitted capacity; no cap exceedance |

| Feedback branch | Decision cost | Oracle cost | Oracle Tax | Approve | Block | Review | Feedback updates |
|---|---:|---:|---:|---:|---:|---:|---:|
| Immediate-after-action oracle | 1,387,589.98 | 1,387,589.98 | 0.00 | 218,078 | 11,091 | 7,047 | 236,216 |
| Blocked labels hidden | 2,875,116.39 | 1,387,589.98 | 1,487,526.41 | 24,525 | 204,644 | 7,047 | 28,216 |
| Blocked labels after 14 days | 2,899,480.41 | 1,387,589.98 | 1,511,890.43 | 20,741 | 208,428 | 7,047 | 198,045 |

| Branch | Correct approvals | Fraud approved | Correct blocks | Genuine blocked | Fraud reviewed | Genuine reviewed | Feedback pending end | Block labels released / pending / hidden |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Immediate-after-action oracle | 211,020 | 7,058 | 1,159 | 9,932 | 458 | 6,589 | 0 | 0 / 0 / 0 |
| Blocked labels hidden | 23,644 | 881 | 7,555 | 197,089 | 239 | 6,808 | 3,161 | 0 / 0 / 204,644 |
| Blocked labels after 14 days | 19,987 | 754 | 7,683 | 200,745 | 238 | 6,809 | 38,003 | 172,931 / 35,497 / 0 |

All three branches have aligned fixed-score metrics and the same action-stream review count. Hidden mode permanently censors blocked outcomes from learner updates; 14-day mode released 172,931 blocked outcomes by the end and left 35,497 blocked outcomes pending. Oracle reveals each label only after the action. The simulation period was examined previously, so the costs are sensitivity results, not independent confirmation.

### Verified model-grid artifact audit — 2026-11-02 (supersedes earlier status snapshots)

A candidate counts as trained only when its current Kaggle version includes the model and prediction artifacts. The following versions were checked via the version output manifests:

| Model / recipe | Version | Artifact evidence | Status |
|---|---:|---|---|
| LightGBM / R0 control | v4 | model, validation and simulation predictions, weight search, thresholds, split summary, run record | Trained; bandit outputs already recorded |
| LightGBM / R1 UID | v3 | model, validation and simulation predictions, weight search, thresholds, split summary, run record | Trained; bandit outputs already recorded |
| LightGBM / R2 full causal | v2 | no model artifacts; version history shows Quick Version; current save/run was rate/session limited | Not trained |
| LightGBM / R3 full causal + identity | v3 | model, validation and simulation predictions, weight search, thresholds, split summary, run record | Trained; bandit outputs now recorded for all three branches |
| XGBoost / R0 control | v6 | model, validation and simulation predictions, weight search, thresholds, split summary, run record | Trained; bandit outputs already recorded |
| XGBoost / R1 UID | v3 | model, validation and simulation predictions, weight search, thresholds, split summary, run record | Trained; bandit outputs already recorded |
| XGBoost / R2 full causal | v2 | model, validation and simulation predictions, weight search, thresholds, split summary, run record | Trained; bandit outputs already recorded |
| XGBoost / R3 full causal + identity | v2 | model, validation and simulation predictions, weight search, thresholds, split summary, run record | Trained; bandit outputs already recorded |
| CatBoost / R0 control | v3 | model, validation and simulation predictions, weight search, thresholds, split summary, run record | Trained; bandit outputs already recorded |
| CatBoost / R1 UID | v3 | API still reports RUNNING; no output manifest yet | Running; don't simulate until artifacts are confirmed |
| CatBoost / R2 full causal | v2 | no model artifacts; version history shows Quick Version | Not trained |
| CatBoost / R3 full causal + identity | v2 | no model artifacts | Not trained |

The completed seed-42 seed-screen coverage is therefore 8 trained supervised model/recipe pairs (including LightGBM R3), with 8 already simulated through all three feedback branches. Missing supervised fits at this audit point: LightGBM R2 and CatBoost R1/R2/R3. The four planned anomaly fits remain unverified/not completed. Kaggle editor reported its maximum batch CPU session count of 5 and rejected a new Save & Run All; its Active Events panel simultaneously showed no active events. Treat that discrepancy as a Kaggle capacity/account-session issue, not as a successful queue. No unidentified session was cancelled.

### Active execution update — 2026-11-02 (Kaggle follow-up)

| Candidate | Current evidence | Disposition |
|---|---|---|
| CatBoost R1 UID seed 42 | Version 3 is a real Save & Run All, still RUNNING after roughly 31 minutes; no output manifest/log visible yet | Keep monitoring; do not cancel while it is doing useful work |
| CatBoost R2 full causal seed 42 | A new Save & Run All was accepted from the editor; session-status API now reports RUNNING | First actual training attempt; await run record and prediction/model artifacts |
| LightGBM R2 full causal seed 42 | Submission attempt failed with Kaggle 'Too many requests' and the UI also reported the five-session CPU cap | Not launched/queued; retry only after a known run completes and Kaggle accepts it |
| CatBoost R3 full causal + identity seed 42 | No verified Save & Run All version yet | Pending |

The editor confirms preprocessing v12 is attached to both inspected R2 notebooks, and the model/recipe/seed values are correct. The training pipeline already exposes a tqdm progress indicator for its class-weight fits. Kaggle's activity panel says zero active events while the API/editor report running sessions and a CPU cap; treat the cap as authoritative for submission attempts. Quick Versions remain invalid as training evidence. Continue checking model-specific status and artifacts, then use available slots for remaining work without claiming rejected jobs are queued.

### Kaggle capacity update — 2026-11-02

- CatBoost R3 full causal + identity seed 42 was prepared in its Kaggle editor with preprocessing v12 attached and the correct config. Save & Run All was rejected by Kaggle with “CPU Session cap reached: maximum batch CPU session count of 5”. No new run version was created; it is not queued.
- Current verifiable active work remains CatBoost R1 UID v3 and CatBoost R2 full causal (new Save & Run All accepted). Both report RUNNING but have no session log/output manifest yet.
- The Kaggle Active Events panel shows zero, and the notebook-search API reports only the two project sessions as RUNNING. Another generic notebook slug has no runs. No identifiable safe session is available to cancel.
- Continue monitoring these two runs, accept more training jobs only when Kaggle permits them, and then launch the matching bandit simulations for valid completed model candidates. R3 and LightGBM R2 remain pending.

### Kaggle execution correction — 2026-11-02, latest direct check

| Candidate | Verified current state | Evidence / interpretation |
|---|---|---|
| CatBoost R1 UID seed 42 | Save & Run All v3 is still RUNNING after 2,885 seconds (48.1 min) | Version-specific log currently shows module loading at 11 seconds, but no model progress/artifacts. This is unusually slow, but Kaggle still reports RUNNING; treat as long-running/opaque, not confirmed failed. |
| CatBoost R2 Full Causal seed 42 | Save & Run All v3 is still RUNNING, listed as started about 14 minutes ago | Version history confirms this is a real run, correcting the previous note that described only v2 Quick Version. No artifacts/results yet. |
| LightGBM R2 / CatBoost R3 | Not running | Their Save & Run All attempts were rejected at Kaggle's five-session CPU cap; Kaggle did not queue them. |

The UI run timer and version history are stronger evidence than stale `last_run_time` metadata. The output API still returns no files for either active CatBoost run. Do not stop either session without evidence of failure; no unrelated session has been cancelled. While they run, no additional training submission has been accepted.

**Progress visibility fix for future CatBoost runs:** local `src/fraudlab/models.py` now asks CatBoost to print a training update every 100 iterations. The outer progress bar still advances when each of the three class-weight fits completes. This change does not affect model parameters or current Kaggle sessions; those already-running notebook versions embed the earlier module snapshot and will not inherit it.

**Correction to progress-logging note:** CatBoost's `fit(..., verbose=False)` had overridden the estimator setting. Both the estimator and fit call now use `verbose=100`, so future notebook versions should emit CatBoost iteration updates every 100 trees. This is progress logging only; no active run is altered, and the update does not add a percent-complete bar inside an individual fit.

### Recheck after progress-logging fix — 2026-11-02

| Candidate | Kaggle status | Elapsed / log | Artifacts |
|---|---|---|---|
| CatBoost R1 UID v3 | RUNNING | 2,960 seconds (~49.3 min); run page timer advances, but the visible log has no new lines after module loading | None listed by session-output API |
| CatBoost R2 Full Causal v3 | RUNNING | Version history still labels it Running; session-output API returns an empty log | None listed by session-output API |

Neither run is accepted as a result yet. No error is visible, so I have not cancelled either. The 100-iteration logging fix is local-only and will apply only after it is included in a later Kaggle Save & Run All version. The next step is to continue monitoring for a completed version or an explicit failure, then recover only the affected notebook; other training launches remain blocked by Kaggle's reported CPU-session cap.

### Runtime check — 2026-11-02, 06:58 IST

| Candidate | Status | Approx. elapsed | New logs/artifacts |
|---|---|---:|---|
| CatBoost R1 UID v3 | RUNNING | 52 min | None reported by Kaggle output API |
| CatBoost R2 Full Causal v3 | RUNNING | 19 min | None reported by Kaggle output API |

A multi-weight CatBoost CPU fit on this dataset can take much longer than LightGBM/XGBoost, but R1's elapsed time with no visible progress is not reassuring. Kaggle has not emitted a failure, so this is classified as an opaque/abnormally slow run rather than a confirmed hang. Continue monitoring the candidate-specific run page; only treat it as complete after artifacts appear.

### CatBoost completion check — 2026-11-02

| Candidate | Status | Runtime | Progress visibility | Result / artifacts |
|---|---|---:|---|---|
| CatBoost / R1 UID / seed 42 / v3 | COMPLETE | 3,697 sec (~61.6 min; pipeline timer 3,672.5 sec) | No tqdm bar or in-fit iteration updates appeared in Kaggle's saved execution log; result JSON appeared near the end. This notebook version predates the local CatBoost verbosity fix. | Model, validation and simulation predictions, weight-search table, thresholds, split summary, and run record are present in Kaggle output manifest. Validation PR-AUC 0.5520; ROC-AUC 0.9036; precision@0.5 0.8258; recall@0.5 0.3979; F1@0.5 0.5370; Brier 0.02101; validation cost 224,838.26. |
| CatBoost / R2 full causal / seed 42 / v3 | RUNNING | — | Left running; no interruption or restart. | Await completion and artifacts. |

**Calibration metric caveat:** R1's reported ECE is effectively zero because isotonic calibration was evaluated on the same validation data used to fit the calibrator. Do not interpret this as an independent calibration result. The fit run used CPU (Kaggle accelerator: None). The local `verbose=100` progress fix cannot change this completed run or the already-running R2 version.

### Training queue check — 2026-11-02 (after CatBoost R1 completion)

| Candidate | Current session state | Training artifacts | Conclusion |
|---|---|---|---|
| CatBoost / R2 full causal / seed 42 / v3 | RUNNING | Not yet checked/present | Only confirmed active model fit; leave running |
| LightGBM / R2 full causal | API says COMPLETE | No output files returned by the output-manifest check | No verified trained model; treat as still needing a real Save & Run All |
| CatBoost / R3 full causal + identity | No active session found | None verified | Pending training |
| Isolation Forest and one-class SVM (transaction and identity variants) | No runs verified | None verified | Four planned anomaly-detection fits remain pending |

### Correction: CatBoost R2 v3 was not a training run — 2026-11-02

The Kaggle R2 Logs page for run `364656916` says “Successfully ran in 7.6s” and “Output 0 B”. Its log contains only Python package SyntaxWarnings and notebook-to-HTML conversion; there are no module-load, training, metric, or save messages. The current notebook has the expected `train_recipe_model_from_artifacts(...)` call in its training cell, but the Kaggle notebook metadata has null execution counts and no cell outputs. Therefore this 7.6-second event did not execute/complete the model-training cell and produced no CatBoost R2 model. The API's `RUNNING` response was stale/inconsistent with the run page. CatBoost R2 remains **not trained**; this entry supersedes the earlier note that v3 was still running. Do not count this quick run as a model result.

### CatBoost R2 GPU training — 2026-11-02

| Item | Verified result |
|---|---|
| Kaggle notebook | `xerxes187/triple-a-grid-catboost-r2-full-causal`, Save & Run All v5 |
| Earlier attempt | v4 enabled the T4 but failed before training because the preprocessing output was not attached; no result counted |
| Input | Preprocessing notebook v12, transaction-only `outputs/preprocessing/transaction/ieee_cis_features.parquet`; attached and found by the run |
| Compute | Tesla T4; CatBoost 1.2.10; GPU devices visible: 2; GPU required, no CPU fallback |
| Split / seed | Existing 48% fit / 12% validation / 40% simulation; seed 42 |
| Progress | tqdm across 3 weighting fits reached 0%, 33%, 67%, 100%; CatBoost iteration logs emitted every 100 iterations |
| Weight search | None / sqrt / balanced; selected on validation PR-AUC per pipeline |
| Training runtime | 268.7 sec model pipeline; Kaggle session completed in about 4m51s |
| Validation PR-AUC | 0.533658 |
| Validation ROC-AUC | 0.901177 |
| Precision / recall / F1 at 0.5 | 0.844011 / 0.366828 / 0.511392 |
| Brier score | 0.021531 |
| Validation decision cost | 224,842.2386 |
| Artifacts | Run record, config, split summary, thresholds, weight-search CSV, CatBoost model + metadata, validation and simulation predictions present in Kaggle v5 output manifest |
| Calibration caveat | ECE 5.3e-18 is not an independent calibration estimate: isotonic calibration was fit and assessed on the same validation period |
| Interpretation | GPU R2 training is complete. Its validation PR-AUC is lower than CatBoost R1's 0.5520, but compare as exploratory because the R1 run used CPU and these are different feature recipes. Simulation-period/bandit evaluation remains the next required stage. |

Kaggle run: https://www.kaggle.com/code/xerxes187/triple-a-grid-catboost-r2-full-causal/log?scriptVersionId=364733746

### CatBoost bandit feedback comparison — 2026-11-02

Completed the two missing seed-42 bandit evaluations. Each Kaggle notebook ran all three feedback branches sequentially with the fixed saved CatBoost score set, the same 236,216-transaction simulation stream, existing costs, and the 3% per-day review cap. The two Kaggle sessions overlapped. Simulation compute was CPU-only (`Accelerator None`); no model was retrained in these runs.

| Candidate | Kaggle version / run | Inputs verified | Runtime | Output |
|---|---|---|---:|---|
| CatBoost R1 UID / seed 42 | v3 / `364755548` | Preprocessing v12 + R1 seed-42 saved predictions and thresholds | 137.5 sec | 56.48 MB; three branch metric files and decision logs |
| CatBoost R2 Full Causal / seed 42 | v3 / `364755992` | Preprocessing v12 + R2 seed-42 GPU-trained saved predictions and thresholds | 143.3 sec | 56.49 MB; three branch metric files and decision logs |

The branch outputs were downloaded under `outputs/bandit/catboost-r1-uid-bandit-run-42/` and `outputs/bandit/catboost-r2-full-causal-bandit-run-42/`. The combined six-row result file is `outputs/bandit/catboost_feedback_comparison_seed42.csv`.

**Acceptance checks:** all six decision logs contain 236,216 rows and their recorded action costs sum to the corresponding metrics cost (within 0.02 points). Each branch made exactly 7,047 reviews; the simulator's exact daily capacity sum is 7,047, with zero per-day cap violations. The review rate is 2.9833% and capacity utilization is 100%. The same score-provider metrics recur across each model's three branches, as expected for fixed predictions. Hidden-block branches schedule/release no blocked labels and mark every blocked label permanently censored. In the 14-day branch, every blocked label is scheduled exactly 14 days after its action; scheduled blocked labels equal blocked actions, and released plus still-pending blocked labels equal that total. Oracle logs show same-day label release and 236,216 post-action feedback updates; the first action uses the initial uniform policy, so its own label is not used before choosing it. Cost sums, row alignment, and all three branch artifacts were verified.

| CatBoost recipe | Feedback branch | PR-AUC | ROC-AUC | Cost | Oracle cost | Oracle Tax | Approve / Review / Block | Approve genuine / fraud missed | Block fraud / genuine falsely declined | Review fraud / genuine | Feedback updates | Block labels released / pending / permanently hidden | Cost vs always approve | Cost vs always block | Cost vs no-review | Cost vs validation-tuned |
|---|---|---:|---:|---:|---:|---:|---|---|---|---|---:|---|---:|---:|---:|---:|
| R1 UID | Immediate-feedback oracle | 0.516372 | 0.894959 | 1,342,547.10 | 1,342,547.10 | 0.00 | 216,662 / 7,047 / 12,507 | 210,117 / 6,545 | 1,642 / 10,865 | 488 / 6,559 | 236,216 | 0 / 0 / 0 | -63,507.08 | -1,731,292.55 | +399,060.89 | +524,954.68 |
| R1 UID | Blocked labels hidden | 0.516372 | 0.894959 | 2,865,390.55 | 1,342,547.10 | 1,522,843.45 | 25,970 / 7,047 / 203,199 | 25,025 / 6,545 | 7,489 / 195,710 | 488 / 6,559 | 29,565 | 0 / 0 / 203,199 | +1,459,336.37 | -208,449.10 | +1,921,904.34 | +2,047,798.13 |
| R1 UID | Blocked labels after 14 days | 0.516372 | 0.894959 | 2,888,490.03 | 1,342,547.10 | 1,545,942.93 | 21,787 / 7,047 / 207,382 | 20,990 / 6,545 | 7,659 / 199,723 | 488 / 6,559 | 197,843 | 172,080 / 35,302 / 0 | +1,482,435.85 | -185,349.62 | +1,945,003.82 | +2,070,897.61 |
| R2 Full Causal | Immediate-feedback oracle | 0.509824 | 0.898446 | 1,369,322.96 | 1,369,322.96 | 0.00 | 217,428 / 7,047 / 11,741 | 210,552 / 6,876 | 1,350 / 10,391 | 449 / 6,598 | 236,216 | 0 / 0 / 0 | -36,731.22 | -1,704,516.70 | +459,705.65 | +562,830.48 |
| R2 Full Causal | Blocked labels hidden | 0.509824 | 0.898446 | 2,867,471.27 | 1,369,322.96 | 1,498,148.31 | 25,641 / 7,047 / 203,528 | 24,716 / 6,876 | 7,509 / 196,019 | 241 / 6,806 | 29,202 | 0 / 0 / 203,528 | +1,461,417.09 | -206,368.38 | +1,957,853.96 | +2,060,978.79 |
| R2 Full Causal | Blocked labels after 14 days | 0.509824 | 0.898446 | 2,896,403.65 | 1,369,322.96 | 1,527,080.69 | 21,246 / 7,047 / 207,923 | 20,467 / 6,876 | 7,663 / 200,260 | 233 / 6,814 | 197,869 | 172,764 / 35,159 / 0 | +1,490,349.46 | -177,436.01 | +1,986,786.34 | +2,089,911.17 |

Cost deltas are `bandit branch cost − baseline cost`; negative means that branch cost less. The shared baselines were: always approve 1,406,054.18; always block 3,073,839.65; no-review score threshold 943,486.21 (R1) / 909,617.31 (R2); validation-tuned threshold 817,592.42 (R1) / 806,492.48 (R2). R1's no-review and tuned baselines differ from R2's because each uses its model's fixed scores/thresholds.

**Interpretation:** the immediate-feedback bandit costs less than always-approve, but still costs more than both no-review and validation-tuned threshold baselines for either model. Closed-loop costs are much higher; both feedback branches spend most actions blocking and falsely decline roughly 196k–200k genuine transactions. This makes the current bandit policy a clear performance bottleneck in this experiment, even with immediate feedback. Oracle Tax measures only the extra cost from delayed/hidden feedback relative to the same bandit receiving feedback immediately after acting; it is not the gap to an optimal policy. These are sensitivity results on the already examined simulation period, not independent confirmation or a basis for selecting a model/preprocessing winner. R1 has higher PR-AUC by 0.00655; R2 has higher ROC-AUC by 0.00349. Neither score-ranking difference makes the current closed-loop policy competitive with the static threshold baselines.

Earlier status notes describing these two bandit runs as pending or running refer to the pre-submission state and are superseded by this verified completion record.

### CatBoost feedback completion and follow-on training — 2026-11-02

| Run | Kaggle version / session | Compute | Input / recipe | Current status |
|---|---|---|---|---|
| CatBoost R1 UID bandit | v3 / 364755548 | CPU | Preprocessing v12 + saved R1 scores | Complete; 137.5 sec; all 3 feedback branches verified |
| CatBoost R2 Full Causal bandit | v3 / 364755992 | CPU | Preprocessing v12 + saved R2 scores | Complete; 143.3 sec; all 3 branches verified |
| LightGBM R2 Full Causal training | v4 / editor session 364759804 | CPU | Preprocessing v12 transaction artifact | Running; tqdm at 0/3 class-weight fits at first live check |
| CatBoost R3 Full Causal + Identity training | v4 / editor session 364759871 | T4 GPU | Preprocessing v12 transaction + identity artifacts | Running; GPU iteration log reached 500 trees in 1m12s, ~2m39s remaining for that fit at first live check |

The R1/R2 bandit outputs, cap checks, branch release checks, baselines and metrics are documented above and saved under `outputs/bandit/`; combined results are in `outputs/bandit/catboost_feedback_comparison_seed42.csv`. Both simulation runs used CPU and overlapped. The training notebooks were saved with Save & Run All; Kaggle's `/log` page initially showed a stale 0 B result, while the live editor showed current progress. Treat the editor's live progress as authoritative until completed artifacts are verified. CatBoost R3's attached v12 output contains both `outputs/preprocessing/transaction/ieee_cis_features.parquet` and `outputs/preprocessing/identity/ieee_cis_features.parquet`; GPU metadata is enabled (`NvidiaTeslaT4`) and the live CatBoost iteration counter confirms training is active.

These two trainings are the remaining named GBE gaps in this batch. After both finish, validate output manifests and metrics, then run each saved score set through the same three feedback branches. Simulation-period results remain exploratory because the final 40% was previously examined.

### Training completion check — 2026-11-02

| Candidate | Kaggle run | Status | Runtime / compute | Verified output artifacts |
|---|---|---|---|---|
| LightGBM / R2 Full Causal / seed 42 | v4 / `364760469` | Complete | 457.1 sec; CPU | Model, model metadata, validation and simulation predictions, thresholds, weight search, split summary, config, candidate metadata, run record |
| CatBoost / R3 Full Causal + Identity / seed 42 | v4 / `364760365` | Complete | 452.2 sec Kaggle session (435.4 sec pipeline); T4 x2 GPU | Model, model metadata, validation and simulation predictions, thresholds, weight search, split summary, config, candidate metadata, run record |

CatBoost R3 validation metrics from the completed run log: PR-AUC 0.539668; ROC-AUC 0.903166; precision 0.869313; recall 0.362389; F1 0.511535 at threshold 0.5; Brier 0.021330; validation decision cost 223,008.84. Its ECE was effectively zero because the isotonic calibrator was fit and evaluated on the same validation period, so that ECE is optimistic and not independent.

Both jobs have completed and their output manifests contain the expected trained-model and prediction artifacts. This closes the training runs only; the matching three-branch bandit simulations for LightGBM R2 and CatBoost R3 have not been run yet. The broader workflow remains paused pending the user's direction.

### Remaining bandit evaluations completed — 2026-11-03

The approved follow-on simulation work is now complete for the trained candidates in scope. Five Kaggle CPU replay runs each evaluated the fixed saved scores under all three feedback branches, using the shared 236,216-row held-out stream, existing costs, and 3% daily review cap. Each Kaggle run completed with saved `metrics.csv`, summary, baseline, plot, and three action logs. All five used seed 42.

| Candidate | Kaggle version / run | Runtime | Accelerator | Result artifacts |
|---|---|---:|---|---|
| LightGBM R2 full causal | v5 / 365000072 | 147.3 sec | CPU | Verified |
| CatBoost R3 full causal + identity | v4 / 365000230 | 138.6 sec | CPU | Verified |
| Isolation Forest R2 full causal | v5 / 365001533 | 145.2 sec | CPU | Verified |
| SGD One-Class SVM R2 full causal | v4 / 365002542 | 134.2 sec | CPU | Verified |
| SGD One-Class SVM R3 full causal + identity | v3 / 365003804 | 141.6 sec | CPU | Verified |

The CatBoost R3 supervised fit used the Kaggle T4 GPU in its earlier training run; GPU is not useful for these replay-only simulations. All completed bandit jobs used CPU. Local consolidated metrics are in `outputs/bandit_remaining_seed42.csv`; the comparison chart is in `outputs/comparison_plots/bandit_remaining_costs_seed42.svg`. The raw detailed action and feedback logs remain available in the corresponding Kaggle notebook outputs.

| Candidate | Fixed-score PR-AUC | Oracle cost | Hidden cost / tax | 14-day cost / tax | Actions (approve / block / review; hidden) | Review-cap check |
|---|---:|---:|---:|---:|---|---|
| LightGBM R2 | 0.483400 | 1,391,049.92 | 2,877,729.17 / 1,486,679.24 | 2,903,323.37 / 1,512,273.45 | 23,912 / 205,257 / 7,047 | 2.9833%; 100%; no exceedance |
| CatBoost R3 + identity | 0.512952 | 1,358,048.20 | 2,867,222.87 / 1,509,174.68 | 2,890,953.28 / 1,532,905.08 | 25,730 / 203,439 / 7,047 | 2.9833%; 100%; no exceedance |
| Isolation Forest R2 | 0.151851 | 1,446,963.19 | 2,881,584.47 / 1,434,621.28 | 2,905,979.15 / 1,459,015.96 | 23,835 / 205,334 / 7,047 | 2.9833%; 100%; no exceedance |
| SGD One-Class SVM R2 | 0.039012 | 1,446,758.98 | 2,883,242.25 / 1,436,483.26 | 2,901,280.50 / 1,454,521.51 | 23,489 / 205,680 / 7,047 | 2.9833%; 100%; no exceedance |
| SGD One-Class SVM R3 + identity | 0.040394 | 1,446,758.98 | 2,883,260.99 / 1,436,502.01 | 2,901,271.27 / 1,454,512.29 | 23,488 / 205,681 / 7,047 | 2.9833%; 100%; no exceedance |

**Interpretation:** CatBoost R3 + identity has the strongest PR-AUC among this five-candidate subset, but its closed-loop cost remains about 2.87–2.89 million points. The immediate-feedback costs are also about 1.36–1.45 million. For all five candidates, both closed-loop branches cost far more than their immediate-feedback oracle; Oracle Tax is about 1.43–1.53 million. The policy still tends to block nearly all transactions when feedback is constrained, falsely declining many genuine transactions. This supports the earlier finding that the current bandit/policy design is the main decision-cost bottleneck; better ranking alone is not enough. Oracle cost is the cost of this same policy with labels immediately available after actions, not the cost of an optimal fraud policy.

All runs used the same costs and review rules. Review counts were below or equal to capacity (6,670–7,047); no per-day cap exceedance was found in the generated logs. In hidden mode, every blocked label remained hidden. In the 14-day branch, blocked feedback releases were 172,631–173,675, with 35,058–35,505 still pending at the simulation horizon. Oracle feedback was post-action for all 236,216 transactions.

**Scope note:** Isolation Forest R3 was not run because no matching trained prediction artifact exists. It is not counted as completed; the current approved batch covered the existing trained candidates and the named GBE follow-on work, not a new training run. Deep learning was explicitly deferred. These are seed-42 sensitivity results on the already examined simulation period, not independent confirmation.
