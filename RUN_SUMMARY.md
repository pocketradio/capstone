# Verified Model and Bandit Results

**Verified scope:** 15 model–preprocessing setups; 45 completed branch results; seed 42. Same 236,216-transaction stream and 3% daily review cap. Results are exploratory because the simulation period was examined previously. Results are ordered by simulation PR-AUC for description only; select models using validation results.

## Published IEEE-CIS Reference Metrics

These are published references, not a claim of a single definitive global SOTA. Threshold-based Precision, Recall, and F1 depend on the chosen operating point.

| Published reference | Evaluation | PR-AUC | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| TEMPLAR-Fraud (hybrid ML framework, 2026) | Internal chronological test; validation-selected threshold | 0.701 | 0.776 | 0.720 | 0.747 |
| TEMPLAR-Fraud (hybrid ML framework, 2026) | Later 10% future-time slice | 0.642 | 0.728 | 0.651 | 0.687 |
| Stacking ensemble (IEEE conference paper, 2025) | Five-seed reported results; mixed reporting points | 0.891 ± 0.005 | 0.950* | 0.847* | 0.856 ± 0.004** |

*For the stacking paper, 0.950 precision and 0.847 recall are reported as Recall@95% Precision. **Its F1 is reported separately; the paper does not establish that F1 uses that same operating point. The paper reports stratified five-fold validation and a separate 20% holdout, which differs from our chronological evaluation. [IEEE publication](https://ieeexplore.ieee.org/abstract/document/11273852) · [Accessible full text](https://www.preprints.org/frontend/manuscript/382f1220a88a61879962108371872653/download_pub).

TEMPLAR-Fraud reports a 70/15/15 chronological train/validation/test split and a separate final 10% same-dataset temporal slice. Its framework combines learned representations, CatBoost, stacking, calibration, and cost-sensitive thresholding; it is not a standalone GBE. Our 48/12/40 split and three-action bandit policy differ, so these figures are context rather than a direct apples-to-apples comparison. [Springer article](https://link.springer.com/article/10.1007/s44248-026-00117-y).
## At a Glance

| Comparison | Model / preprocessing | Result |
|---|---|---:|
| Highest simulation PR-AUC | CatBoost · R0 Control | 0.519628 |
| Lowest hidden-label cost | LightGBM · R0 Control | 2,862,886.74 |
| Lowest 14-day cost | CatBoost · R1 UID | 2,888,490.03 |
| Lowest immediate-oracle cost | CatBoost · R0 Control | 1,336,287.86 |
| Lowest simulation PR-AUC | SGD One-Class SVM · R2 Full Causal | 0.039012 |

**Published comparison reference:** IEEE-CIS stacking ensemble, reported PR-AUC 0.891 ± 0.005 and ROC-AUC 0.918 ± 0.003. The paper reports stratified five-fold validation and a separate 20% holdout; our chronological split differs, so numeric gaps are context, not a like-for-like ranking. [IEEE paper](https://ieeexplore.ieee.org/abstract/document/11273852) · [Preprint with reported split details](https://www.preprints.org/frontend/manuscript/382f1220a88a61879962108371872653/download_pub).

`A/B/R` = approve/block/review. `A✓/miss` = genuine approvals correct/fraud approvals missed. `B✓/false` = fraud blocks correct/genuine transactions wrongly blocked. `R F/G` = reviewed fraud/genuine. `ΔPR/ΔROC` = our score minus the published reference. Within cells, short bullet facts are separated by a centered dot for compact scanning.

## Blocked Labels Hidden

| ML model | Preprocessing | Result | Difference vs. SOTA | RL model metrics | Oracle Tax | Action counts and correctness | Feedback counts | Runtime / seed / run ID |
|---|---|---|---|---|---|---|---|---|
| CatBoost | R0 Control | • PR-AUC: 0.519628 · • ROC-AUC: 0.893092 | • ΔPR-AUC: -0.3714 · • ΔROC-AUC: -0.0249 | • Cost: 2,867,884.41 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,531,596.54 | • A/B/R actions: 25,374/203,795/7,047 · • Approve correct/missed: 24,458/916 · • Block correct/false decline: 7,516/196,279 · • Review fraud/genuine: 243/6,804 | • Labels received: 29,001 · • Block labels hidden: 203,795 | • Runtime: 147s · • Seed: 42 · • Run ID: v2 / 364680035 |
| CatBoost | R1 UID | • PR-AUC: 0.516372 · • ROC-AUC: 0.894959 | • ΔPR-AUC: -0.3746 · • ΔROC-AUC: -0.0230 | • Cost: 2,865,390.55 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,522,843.45 | • A/B/R actions: 25,970/203,199/7,047 · • Approve correct/missed: 25,025/945 · • Block correct/false decline: 7,489/195,710 · • Review fraud/genuine: 241/6,806 | • Labels received: 29,565 · • Block labels hidden: 203,199 | • Runtime: 137.5s · • Seed: 42 · • Run ID: v3 / 364755548 |
| CatBoost | R3 Full Causal + Identity | • PR-AUC: 0.512952 · • ROC-AUC: 0.899728 | • ΔPR-AUC: -0.3780 · • ΔROC-AUC: -0.0183 | • Cost: 2,867,222.87 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,509,174.68 | • A/B/R actions: 25,730/203,439/7,047 · • Approve correct/missed: 24,796/934 · • Block correct/false decline: 7,500/195,939 · • Review fraud/genuine: 241/6,806 | • Labels received: 29,306 · • Block labels hidden: 203,439 | • Runtime: 138.6s · • Seed: 42 · • Run ID: v4 / 365000230 |
| CatBoost | R2 Full Causal | • PR-AUC: 0.509824 · • ROC-AUC: 0.898446 | • ΔPR-AUC: -0.3812 · • ΔROC-AUC: -0.0196 | • Cost: 2,867,471.27 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,498,148.31 | • A/B/R actions: 25,641/203,528/7,047 · • Approve correct/missed: 24,716/925 · • Block correct/false decline: 7,509/196,019 · • Review fraud/genuine: 241/6,806 | • Labels received: 29,202 · • Block labels hidden: 203,528 | • Runtime: 143.3s · • Seed: 42 · • Run ID: v3 / 364755992 |
| XGBoost | R3 Full Causal + Identity | • PR-AUC: 0.501973 · • ROC-AUC: 0.886044 | • ΔPR-AUC: -0.3890 · • ΔROC-AUC: -0.0320 | • Cost: 2,872,110.16 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,498,532.85 | • A/B/R actions: 24,957/204,212/7,047 · • Approve correct/missed: 24,053/904 · • Block correct/false decline: 7,528/196,684 · • Review fraud/genuine: 243/6,804 | • Labels received: 28,609 · • Block labels hidden: 204,212 | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: v2 / 364677034 |
| XGBoost | R2 Full Causal | • PR-AUC: 0.497258 · • ROC-AUC: 0.878826 | • ΔPR-AUC: -0.3937 · • ΔROC-AUC: -0.0392 | • Cost: 2,873,581.21 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,492,128.87 | • A/B/R actions: 24,887/204,282/7,047 · • Approve correct/missed: 23,982/905 · • Block correct/false decline: 7,528/196,754 · • Review fraud/genuine: 242/6,805 | • Labels received: 28,532 · • Block labels hidden: 204,282 | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: v5 / 364676871 |
| XGBoost | R1 UID | • PR-AUC: 0.496087 · • ROC-AUC: 0.879226 | • ΔPR-AUC: -0.3949 · • ΔROC-AUC: -0.0388 | • Cost: 2,872,167.21 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,491,222.24 | • A/B/R actions: 24,993/204,176/7,047 · • Approve correct/missed: 24,085/908 · • Block correct/false decline: 7,524/196,652 · • Review fraud/genuine: 243/6,804 | • Labels received: 28,671 · • Block labels hidden: 204,176 | • Runtime: 144s · • Seed: 42 · • Run ID: v2 / 364674309 |
| XGBoost | R0 Control | • PR-AUC: 0.492914 · • ROC-AUC: 0.878792 | • ΔPR-AUC: -0.3981 · • ΔROC-AUC: -0.0392 | • Cost: 2,872,664.97 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,485,218.21 | • A/B/R actions: 24,898/204,271/7,047 · • Approve correct/missed: 23,993/905 · • Block correct/false decline: 7,527/196,744 · • Review fraud/genuine: 243/6,804 | • Labels received: 28,627 · • Block labels hidden: 204,271 | • Runtime: 119s · • Seed: 42 · • Run ID: v3 / 364672524 |
| LightGBM | R0 Control | • PR-AUC: 0.485809 · • ROC-AUC: 0.866556 | • ΔPR-AUC: -0.4052 · • ΔROC-AUC: -0.0514 | • Cost: 2,862,886.74 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,482,042.71 | • A/B/R actions: 26,364/202,805/7,047 · • Approve correct/missed: 25,414/950 · • Block correct/false decline: 7,486/195,319 · • Review fraud/genuine: 239/6,808 | • Labels received: 29,801 · • Block labels hidden: 202,805 | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: bandit run 364678364 |
| LightGBM | R1 UID | • PR-AUC: 0.485435 · • ROC-AUC: 0.867412 | • ΔPR-AUC: -0.4056 · • ΔROC-AUC: -0.0506 | • Cost: 2,870,297.06 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,479,960.51 | • A/B/R actions: 25,389/203,780/7,047 · • Approve correct/missed: 24,470/919 · • Block correct/false decline: 7,517/196,263 · • Review fraud/genuine: 239/6,808 | • Labels received: 29,069 · • Block labels hidden: 203,780 | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: bandit run 364671573 |
| LightGBM | R3 Full Causal + Identity | • PR-AUC: 0.484229 · • ROC-AUC: 0.866877 | • ΔPR-AUC: -0.4068 · • ΔROC-AUC: -0.0511 | • Cost: 2,875,116.39 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,487,526.41 | • A/B/R actions: 24,525/204,644/7,047 · • Approve correct/missed: 23,644/881 · • Block correct/false decline: 7,555/197,089 · • Review fraud/genuine: 239/6,808 | • Labels received: 28,216 · • Block labels hidden: 204,644 | • Runtime: 136.7s · • Seed: 42 · • Run ID: v1 / 364708967 |
| LightGBM | R2 Full Causal | • PR-AUC: 0.483400 · • ROC-AUC: 0.866211 | • ΔPR-AUC: -0.4076 · • ΔROC-AUC: -0.0518 | • Cost: 2,877,729.17 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,486,679.24 | • A/B/R actions: 23,912/205,257/7,047 · • Approve correct/missed: 23,048/864 · • Block correct/false decline: 7,570/197,687 · • Review fraud/genuine: 241/6,806 | • Labels received: 27,573 · • Block labels hidden: 205,257 | • Runtime: 147.3s · • Seed: 42 · • Run ID: v5 / 365000072 |
| Isolation Forest | R2 Full Causal | • PR-AUC: 0.151851 · • ROC-AUC: 0.769175 | • ΔPR-AUC: -0.7391 · • ΔROC-AUC: -0.1488 | • Cost: 2,881,584.47 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,434,621.28 | • A/B/R actions: 23,835/205,334/7,047 · • Approve correct/missed: 22,937/898 · • Block correct/false decline: 7,536/197,798 · • Review fraud/genuine: 241/6,806 | • Labels received: 27,622 · • Block labels hidden: 205,334 | • Runtime: 145.2s · • Seed: 42 · • Run ID: v5 / 365001533 |
| SGD One-Class SVM | R3 Full Causal + Identity | • PR-AUC: 0.040394 · • ROC-AUC: 0.517606 | • ΔPR-AUC: -0.8506 · • ΔROC-AUC: -0.4004 | • Cost: 2,883,261.00 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,436,502.01 | • A/B/R actions: 23,488/205,681/7,047 · • Approve correct/missed: 22,601/887 · • Block correct/false decline: 7,543/198,138 · • Review fraud/genuine: 245/6,802 | • Labels received: 27,317 · • Block labels hidden: 205,681 | • Runtime: 141.6s · • Seed: 42 · • Run ID: v3 / 365003804 |
| SGD One-Class SVM | R2 Full Causal | • PR-AUC: 0.039012 · • ROC-AUC: 0.505566 | • ΔPR-AUC: -0.8520 · • ΔROC-AUC: -0.4124 | • Cost: 2,883,242.25 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,436,483.26 | • A/B/R actions: 23,489/205,680/7,047 · • Approve correct/missed: 22,602/887 · • Block correct/false decline: 7,543/198,137 · • Review fraud/genuine: 245/6,802 | • Labels received: 27,316 · • Block labels hidden: 205,680 | • Runtime: 134.2s · • Seed: 42 · • Run ID: v4 / 365002542 |


## Blocked Labels After 14 Days

| ML model | Preprocessing | Result | Difference vs. SOTA | RL model metrics | Oracle Tax | Action counts and correctness | Feedback counts | Runtime / seed / run ID |
|---|---|---|---|---|---|---|---|---|
| CatBoost | R0 Control | • PR-AUC: 0.519628 · • ROC-AUC: 0.893092 | • ΔPR-AUC: -0.3714 · • ΔROC-AUC: -0.0249 | • Cost: 2,889,986.35 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,553,698.49 | • A/B/R actions: 21,293/207,876/7,047 · • Approve correct/missed: 20,514/779 · • Block correct/false decline: 7,668/200,208 · • Review fraud/genuine: 228/6,819 | • Labels received: 197,925 · • Block labels delayed 14 days | • Runtime: 147s · • Seed: 42 · • Run ID: v2 / 364680035 |
| CatBoost | R1 UID | • PR-AUC: 0.516372 · • ROC-AUC: 0.894959 | • ΔPR-AUC: -0.3746 · • ΔROC-AUC: -0.0230 | • Cost: 2,888,490.03 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,545,942.93 | • A/B/R actions: 21,787/207,382/7,047 · • Approve correct/missed: 20,990/797 · • Block correct/false decline: 7,659/199,723 · • Review fraud/genuine: 219/6,828 | • Labels received: 197,843 · • Block labels delayed 14 days | • Runtime: 137.5s · • Seed: 42 · • Run ID: v3 / 364755548 |
| CatBoost | R3 Full Causal + Identity | • PR-AUC: 0.512952 · • ROC-AUC: 0.899728 | • ΔPR-AUC: -0.3780 · • ΔROC-AUC: -0.0183 | • Cost: 2,890,953.28 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,532,905.08 | • A/B/R actions: 21,480/207,689/7,047 · • Approve correct/missed: 20,697/783 · • Block correct/false decline: 7,670/200,019 · • Review fraud/genuine: 222/6,825 | • Labels received: 197,884 · • Block labels delayed 14 days | • Runtime: 138.6s · • Seed: 42 · • Run ID: v4 / 365000230 |
| CatBoost | R2 Full Causal | • PR-AUC: 0.509824 · • ROC-AUC: 0.898446 | • ΔPR-AUC: -0.3812 · • ΔROC-AUC: -0.0196 | • Cost: 2,896,403.65 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,527,080.69 | • A/B/R actions: 21,246/207,923/7,047 · • Approve correct/missed: 20,467/779 · • Block correct/false decline: 7,663/200,260 · • Review fraud/genuine: 233/6,814 | • Labels received: 197,869 · • Block labels delayed 14 days | • Runtime: 143.3s · • Seed: 42 · • Run ID: v3 / 364755992 |
| XGBoost | R3 Full Causal + Identity | • PR-AUC: 0.501973 · • ROC-AUC: 0.886044 | • ΔPR-AUC: -0.3890 · • ΔROC-AUC: -0.0320 | • Cost: 2,893,846.14 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,520,268.83 | • A/B/R actions: 21,132/208,037/7,047 · • Approve correct/missed: 20,359/773 · • Block correct/false decline: 7,672/200,365 · • Review fraud/genuine: 230/6,817 | • Labels received: 197,979 · • Block labels delayed 14 days | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: v2 / 364677034 |
| XGBoost | R2 Full Causal | • PR-AUC: 0.497258 · • ROC-AUC: 0.878826 | • ΔPR-AUC: -0.3937 · • ΔROC-AUC: -0.0392 | • Cost: 2,895,841.61 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,514,389.27 | • A/B/R actions: 21,172/207,997/7,047 · • Approve correct/missed: 20,396/776 · • Block correct/false decline: 7,668/200,329 · • Review fraud/genuine: 231/6,816 | • Labels received: 197,911 · • Block labels delayed 14 days | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: v5 / 364676871 |
| XGBoost | R1 UID | • PR-AUC: 0.496087 · • ROC-AUC: 0.879226 | • ΔPR-AUC: -0.3949 · • ΔROC-AUC: -0.0388 | • Cost: 2,895,185.93 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,514,240.96 | • A/B/R actions: 21,067/208,102/7,047 · • Approve correct/missed: 20,294/773 · • Block correct/false decline: 7,678/200,424 · • Review fraud/genuine: 224/6,823 | • Labels received: 197,950 · • Block labels delayed 14 days | • Runtime: 144s · • Seed: 42 · • Run ID: v2 / 364674309 |
| XGBoost | R0 Control | • PR-AUC: 0.492914 · • ROC-AUC: 0.878792 | • ΔPR-AUC: -0.3981 · • ΔROC-AUC: -0.0392 | • Cost: 2,895,581.86 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,508,135.11 | • A/B/R actions: 21,054/208,115/7,047 · • Approve correct/missed: 20,281/773 · • Block correct/false decline: 7,678/200,437 · • Review fraud/genuine: 224/6,823 | • Labels received: 197,952 · • Block labels delayed 14 days | • Runtime: 119s · • Seed: 42 · • Run ID: v3 / 364672524 |
| LightGBM | R0 Control | • PR-AUC: 0.485809 · • ROC-AUC: 0.866556 | • ΔPR-AUC: -0.4052 · • ΔROC-AUC: -0.0514 | • Cost: 2,893,664.29 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,512,820.27 | • A/B/R actions: 21,290/207,879/7,047 · • Approve correct/missed: 20,509/781 · • Block correct/false decline: 7,668/200,211 · • Review fraud/genuine: 226/6,821 | • Labels received: 197,929 · • Block labels delayed 14 days | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: bandit run 364678364 |
| LightGBM | R1 UID | • PR-AUC: 0.485435 · • ROC-AUC: 0.867412 | • ΔPR-AUC: -0.4056 · • ΔROC-AUC: -0.0506 | • Cost: 2,895,521.55 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,505,185.01 | • A/B/R actions: 21,287/207,882/7,047 · • Approve correct/missed: 20,505/782 · • Block correct/false decline: 7,672/200,210 · • Review fraud/genuine: 221/6,826 | • Labels received: 197,927 · • Block labels delayed 14 days | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: bandit run 364671573 |
| LightGBM | R3 Full Causal + Identity | • PR-AUC: 0.484229 · • ROC-AUC: 0.866877 | • ΔPR-AUC: -0.4068 · • ΔROC-AUC: -0.0511 | • Cost: 2,899,480.41 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,511,890.43 | • A/B/R actions: 20,741/208,428/7,047 · • Approve correct/missed: 19,987/754 · • Block correct/false decline: 7,683/200,745 · • Review fraud/genuine: 238/6,809 | • Labels received: 198,045 · • Block labels delayed 14 days | • Runtime: 136.7s · • Seed: 42 · • Run ID: v1 / 364708967 |
| LightGBM | R2 Full Causal | • PR-AUC: 0.483400 · • ROC-AUC: 0.866211 | • ΔPR-AUC: -0.4076 · • ΔROC-AUC: -0.0518 | • Cost: 2,903,323.37 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,512,273.45 | • A/B/R actions: 20,060/209,109/7,047 · • Approve correct/missed: 19,331/729 · • Block correct/false decline: 7,709/201,400 · • Review fraud/genuine: 237/6,810 | • Labels received: 198,027 · • Block labels delayed 14 days | • Runtime: 147.3s · • Seed: 42 · • Run ID: v5 / 365000072 |
| Isolation Forest | R2 Full Causal | • PR-AUC: 0.151851 · • ROC-AUC: 0.769175 | • ΔPR-AUC: -0.7391 · • ΔROC-AUC: -0.1488 | • Cost: 2,905,979.15 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,459,015.96 | • A/B/R actions: 20,217/208,952/7,047 · • Approve correct/missed: 19,464/753 · • Block correct/false decline: 7,676/201,276 · • Review fraud/genuine: 246/6,801 | • Labels received: 198,004 · • Block labels delayed 14 days | • Runtime: 145.2s · • Seed: 42 · • Run ID: v5 / 365001533 |
| SGD One-Class SVM | R3 Full Causal + Identity | • PR-AUC: 0.040394 · • ROC-AUC: 0.517606 | • ΔPR-AUC: -0.8506 · • ΔROC-AUC: -0.4004 | • Cost: 2,901,271.27 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,454,512.29 | • A/B/R actions: 20,229/208,940/7,047 · • Approve correct/missed: 19,475/754 · • Block correct/false decline: 7,674/201,266 · • Review fraud/genuine: 247/6,800 | • Labels received: 198,050 · • Block labels delayed 14 days | • Runtime: 141.6s · • Seed: 42 · • Run ID: v3 / 365003804 |
| SGD One-Class SVM | R2 Full Causal | • PR-AUC: 0.039012 · • ROC-AUC: 0.505566 | • ΔPR-AUC: -0.8520 · • ΔROC-AUC: -0.4124 | • Cost: 2,901,280.50 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 1,454,521.51 | • A/B/R actions: 20,231/208,938/7,047 · • Approve correct/missed: 19,476/755 · • Block correct/false decline: 7,673/201,265 · • Review fraud/genuine: 247/6,800 | • Labels received: 198,050 · • Block labels delayed 14 days | • Runtime: 134.2s · • Seed: 42 · • Run ID: v4 / 365002542 |


## Immediate-Feedback Oracle

| ML model | Preprocessing | Result | Difference vs. SOTA | RL model metrics | Oracle Tax | Action counts and correctness | Feedback counts | Runtime / seed / run ID |
|---|---|---|---|---|---|---|---|---|
| CatBoost | R0 Control | • PR-AUC: 0.519628 · • ROC-AUC: 0.893092 | • ΔPR-AUC: -0.3714 · • ΔROC-AUC: -0.0249 | • Cost: 1,336,287.86 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 216,464/12,705/7,047 · • Approve correct/missed: 209,998/6,466 · • Block correct/false decline: 1,736/10,969 · • Review fraud/genuine: 473/6,574 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 147s · • Seed: 42 · • Run ID: v2 / 364680035 |
| CatBoost | R1 UID | • PR-AUC: 0.516372 · • ROC-AUC: 0.894959 | • ΔPR-AUC: -0.3746 · • ΔROC-AUC: -0.0230 | • Cost: 1,342,547.10 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 216,662/12,507/7,047 · • Approve correct/missed: 210,117/6,545 · • Block correct/false decline: 1,642/10,865 · • Review fraud/genuine: 488/6,559 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 137.5s · • Seed: 42 · • Run ID: v3 / 364755548 |
| CatBoost | R3 Full Causal + Identity | • PR-AUC: 0.512952 · • ROC-AUC: 0.899728 | • ΔPR-AUC: -0.3780 · • ΔROC-AUC: -0.0183 | • Cost: 1,358,048.20 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 216,780/12,389/7,047 · • Approve correct/missed: 210,128/6,652 · • Block correct/false decline: 1,575/10,814 · • Review fraud/genuine: 448/6,599 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 138.6s · • Seed: 42 · • Run ID: v4 / 365000230 |
| CatBoost | R2 Full Causal | • PR-AUC: 0.509824 · • ROC-AUC: 0.898446 | • ΔPR-AUC: -0.3812 · • ΔROC-AUC: -0.0196 | • Cost: 1,369,322.96 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 217,428/11,741/7,047 · • Approve correct/missed: 210,552/6,876 · • Block correct/false decline: 1,350/10,391 · • Review fraud/genuine: 449/6,598 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 143.3s · • Seed: 42 · • Run ID: v3 / 364755992 |
| XGBoost | R3 Full Causal + Identity | • PR-AUC: 0.501973 · • ROC-AUC: 0.886044 | • ΔPR-AUC: -0.3890 · • ΔROC-AUC: -0.0320 | • Cost: 1,373,577.31 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 217,414/11,755/7,047 · • Approve correct/missed: 210,508/6,906 · • Block correct/false decline: 1,330/10,425 · • Review fraud/genuine: 439/6,608 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: v2 / 364677034 |
| XGBoost | R2 Full Causal | • PR-AUC: 0.497258 · • ROC-AUC: 0.878826 | • ΔPR-AUC: -0.3937 · • ΔROC-AUC: -0.0392 | • Cost: 1,381,452.34 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 217,705/11,464/7,047 · • Approve correct/missed: 210,689/7,016 · • Block correct/false decline: 1,213/10,251 · • Review fraud/genuine: 446/6,601 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: v5 / 364676871 |
| XGBoost | R1 UID | • PR-AUC: 0.496087 · • ROC-AUC: 0.879226 | • ΔPR-AUC: -0.3949 · • ΔROC-AUC: -0.0388 | • Cost: 1,380,944.97 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 217,664/11,505/7,047 · • Approve correct/missed: 210,654/7,010 · • Block correct/false decline: 1,228/10,277 · • Review fraud/genuine: 437/6,610 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 144s · • Seed: 42 · • Run ID: v2 / 364674309 |
| XGBoost | R0 Control | • PR-AUC: 0.492914 · • ROC-AUC: 0.878792 | • ΔPR-AUC: -0.3981 · • ΔROC-AUC: -0.0392 | • Cost: 1,387,446.75 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 217,850/11,319/7,047 · • Approve correct/missed: 210,768/7,082 · • Block correct/false decline: 1,152/10,167 · • Review fraud/genuine: 441/6,606 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 119s · • Seed: 42 · • Run ID: v3 / 364672524 |
| LightGBM | R0 Control | • PR-AUC: 0.485809 · • ROC-AUC: 0.866556 | • ΔPR-AUC: -0.4052 · • ΔROC-AUC: -0.0514 | • Cost: 1,380,844.03 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 217,765/11,404/7,047 · • Approve correct/missed: 210,794/6,971 · • Block correct/false decline: 1,258/10,146 · • Review fraud/genuine: 446/6,601 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: bandit run 364678364 |
| LightGBM | R1 UID | • PR-AUC: 0.485435 · • ROC-AUC: 0.867412 | • ΔPR-AUC: -0.4056 · • ΔROC-AUC: -0.0506 | • Cost: 1,390,336.54 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 218,034/11,135/7,047 · • Approve correct/missed: 210,964/7,070 · • Block correct/false decline: 1,151/9,984 · • Review fraud/genuine: 454/6,593 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: runtime not recorded · • Seed: 42 · • Run ID: bandit run 364671573 |
| LightGBM | R3 Full Causal + Identity | • PR-AUC: 0.484229 · • ROC-AUC: 0.866877 | • ΔPR-AUC: -0.4068 · • ΔROC-AUC: -0.0511 | • Cost: 1,387,589.98 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 218,078/11,091/7,047 · • Approve correct/missed: 211,020/7,058 · • Block correct/false decline: 1,159/9,932 · • Review fraud/genuine: 458/6,589 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 136.7s · • Seed: 42 · • Run ID: v1 / 364708967 |
| LightGBM | R2 Full Causal | • PR-AUC: 0.483400 · • ROC-AUC: 0.866211 | • ΔPR-AUC: -0.4076 · • ΔROC-AUC: -0.0518 | • Cost: 1,391,049.92 · • Reviews: 7,047 · • Review rate / cap: 2.983% / 100% | • 0.00 | • A/B/R actions: 218,044/11,125/7,047 · • Approve correct/missed: 210,937/7,107 · • Block correct/false decline: 1,128/9,997 · • Review fraud/genuine: 440/6,607 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 147.3s · • Seed: 42 · • Run ID: v5 / 365000072 |
| Isolation Forest | R2 Full Causal | • PR-AUC: 0.151851 · • ROC-AUC: 0.769175 | • ΔPR-AUC: -0.7391 · • ΔROC-AUC: -0.1488 | • Cost: 1,446,963.19 · • Reviews: 6,845 · • Review rate / cap: 2.898% / 97.13% | • 0.00 | • A/B/R actions: 222,399/6,972/6,845 · • Approve correct/missed: 214,253/8,146 · • Block correct/false decline: 245/6,727 · • Review fraud/genuine: 284/6,561 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 145.2s · • Seed: 42 · • Run ID: v5 / 365001533 |
| SGD One-Class SVM | R3 Full Causal + Identity | • PR-AUC: 0.040394 · • ROC-AUC: 0.517606 | • ΔPR-AUC: -0.8506 · • ΔROC-AUC: -0.4004 | • Cost: 1,446,758.98 · • Reviews: 6,670 · • Review rate / cap: 2.824% / 94.65% | • 0.00 | • A/B/R actions: 222,744/6,802/6,670 · • Approve correct/missed: 214,577/8,167 · • Block correct/false decline: 237/6,565 · • Review fraud/genuine: 271/6,399 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 141.6s · • Seed: 42 · • Run ID: v3 / 365003804 |
| SGD One-Class SVM | R2 Full Causal | • PR-AUC: 0.039012 · • ROC-AUC: 0.505566 | • ΔPR-AUC: -0.8520 · • ΔROC-AUC: -0.4124 | • Cost: 1,446,758.98 · • Reviews: 6,670 · • Review rate / cap: 2.824% / 94.65% | • 0.00 | • A/B/R actions: 222,744/6,802/6,670 · • Approve correct/missed: 214,577/8,167 · • Block correct/false decline: 237/6,565 · • Review fraud/genuine: 271/6,399 | • Labels received: 236,216 · • Released after action · • Pending: 0 | • Runtime: 134.2s · • Seed: 42 · • Run ID: v4 / 365002542 |


## Overall and Per-Decision Fraud Metrics

P / R / F1 = precision / recall / F1, as percentages. Overall triage treats Block and Review as fraud flagged, and Approve as not flagged. Per-action figures are one-vs-rest: precision = fraud assigned to that action / all transactions assigned to it; recall = fraud assigned to it / all fraud in the simulation stream. F1 is the harmonic mean.

### Blocked Labels Hidden

| ML model | Preprocessing | Overall flagged P / R / F1 | Approve P / R / F1 | Review P / R / F1 | Block P / R / F1 |
|---|---|---:|---:|---:|---:|
| CatBoost | R0 Control | 3.68% / 89.44% / 7.07% | 3.61% / 10.56% / 5.38% | 3.45% / 2.80% / 3.09% | 3.69% / 86.64% / 7.07% |
| CatBoost | R1 UID | 3.68% / 89.11% / 7.06% | 3.64% / 10.89% / 5.46% | 3.42% / 2.78% / 3.07% | 3.69% / 86.33% / 7.07% |
| CatBoost | R3 Full Causal + Identity | 3.68% / 89.23% / 7.06% | 3.63% / 10.77% / 5.43% | 3.42% / 2.78% / 3.07% | 3.69% / 86.46% / 7.07% |
| CatBoost | R2 Full Causal | 3.68% / 89.34% / 7.07% | 3.61% / 10.66% / 5.39% | 3.42% / 2.78% / 3.07% | 3.69% / 86.56% / 7.08% |
| XGBoost | R3 Full Causal + Identity | 3.68% / 89.58% / 7.07% | 3.62% / 10.42% / 5.38% | 3.45% / 2.80% / 3.09% | 3.69% / 86.78% / 7.07% |
| XGBoost | R2 Full Causal | 3.68% / 89.57% / 7.06% | 3.64% / 10.43% / 5.39% | 3.43% / 2.79% / 3.08% | 3.69% / 86.78% / 7.07% |
| XGBoost | R1 UID | 3.68% / 89.53% / 7.06% | 3.63% / 10.47% / 5.39% | 3.45% / 2.80% / 3.09% | 3.69% / 86.73% / 7.07% |
| XGBoost | R0 Control | 3.68% / 89.57% / 7.06% | 3.63% / 10.43% / 5.39% | 3.45% / 2.80% / 3.09% | 3.68% / 86.77% / 7.07% |
| LightGBM | R0 Control | 3.68% / 89.05% / 7.07% | 3.60% / 10.95% / 5.42% | 3.39% / 2.76% / 3.04% | 3.69% / 86.29% / 7.08% |
| LightGBM | R1 UID | 3.68% / 89.41% / 7.07% | 3.62% / 10.59% / 5.40% | 3.39% / 2.76% / 3.04% | 3.69% / 86.65% / 7.08% |
| LightGBM | R3 Full Causal + Identity | 3.68% / 89.84% / 7.07% | 3.59% / 10.16% / 5.31% | 3.39% / 2.76% / 3.04% | 3.69% / 87.09% / 7.08% |
| LightGBM | R2 Full Causal | 3.68% / 90.04% / 7.07% | 3.61% / 9.96% / 5.30% | 3.42% / 2.78% / 3.07% | 3.69% / 87.26% / 7.08% |
| Isolation Forest | R2 Full Causal | 3.66% / 89.65% / 7.04% | 3.77% / 10.35% / 5.52% | 3.42% / 2.78% / 3.07% | 3.67% / 86.87% / 7.04% |
| SGD One-Class SVM | R3 Full Causal + Identity | 3.66% / 89.78% / 7.04% | 3.78% / 10.22% / 5.52% | 3.48% / 2.82% / 3.12% | 3.67% / 86.95% / 7.04% |
| SGD One-Class SVM | R2 Full Causal | 3.66% / 89.78% / 7.04% | 3.78% / 10.22% / 5.52% | 3.48% / 2.82% / 3.12% | 3.67% / 86.95% / 7.04% |

### Blocked Labels After 14 Days

| ML model | Preprocessing | Overall flagged P / R / F1 | Approve P / R / F1 | Review P / R / F1 | Block P / R / F1 |
|---|---|---:|---:|---:|---:|
| CatBoost | R0 Control | 3.67% / 91.02% / 7.06% | 3.66% / 8.98% / 5.20% | 3.24% / 2.63% / 2.90% | 3.69% / 88.39% / 7.08% |
| CatBoost | R1 UID | 3.67% / 90.81% / 7.06% | 3.66% / 9.19% / 5.23% | 3.11% / 2.52% / 2.79% | 3.69% / 88.29% / 7.09% |
| CatBoost | R3 Full Causal + Identity | 3.68% / 90.97% / 7.07% | 3.65% / 9.03% / 5.19% | 3.15% / 2.56% / 2.82% | 3.69% / 88.41% / 7.09% |
| CatBoost | R2 Full Causal | 3.67% / 91.02% / 7.06% | 3.67% / 8.98% / 5.21% | 3.31% / 2.69% / 2.96% | 3.69% / 88.33% / 7.08% |
| XGBoost | R3 Full Causal + Identity | 3.67% / 91.09% / 7.06% | 3.66% / 8.91% / 5.19% | 3.26% / 2.65% / 2.93% | 3.69% / 88.44% / 7.08% |
| XGBoost | R2 Full Causal | 3.67% / 91.05% / 7.06% | 3.67% / 8.95% / 5.20% | 3.28% / 2.66% / 2.94% | 3.69% / 88.39% / 7.08% |
| XGBoost | R1 UID | 3.67% / 91.09% / 7.06% | 3.67% / 8.91% / 5.20% | 3.18% / 2.58% / 2.85% | 3.69% / 88.51% / 7.08% |
| XGBoost | R0 Control | 3.67% / 91.09% / 7.06% | 3.67% / 8.91% / 5.20% | 3.18% / 2.58% / 2.85% | 3.69% / 88.51% / 7.08% |
| LightGBM | R0 Control | 3.67% / 91.00% / 7.06% | 3.67% / 9.00% / 5.21% | 3.21% / 2.61% / 2.87% | 3.69% / 88.39% / 7.08% |
| LightGBM | R1 UID | 3.67% / 90.99% / 7.06% | 3.67% / 9.01% / 5.22% | 3.14% / 2.55% / 2.81% | 3.69% / 88.44% / 7.09% |
| LightGBM | R3 Full Causal + Identity | 3.68% / 91.31% / 7.07% | 3.64% / 8.69% / 5.13% | 3.38% / 2.74% / 3.03% | 3.69% / 88.56% / 7.08% |
| LightGBM | R2 Full Causal | 3.68% / 91.60% / 7.07% | 3.63% / 8.40% / 5.07% | 3.36% / 2.73% / 3.01% | 3.69% / 88.86% / 7.08% |
| Isolation Forest | R2 Full Causal | 3.67% / 91.32% / 7.05% | 3.72% / 8.68% / 5.21% | 3.49% / 2.84% / 3.13% | 3.67% / 88.48% / 7.05% |
| SGD One-Class SVM | R3 Full Causal + Identity | 3.67% / 91.31% / 7.05% | 3.73% / 8.69% / 5.22% | 3.51% / 2.85% / 3.14% | 3.67% / 88.46% / 7.05% |
| SGD One-Class SVM | R2 Full Causal | 3.67% / 91.30% / 7.05% | 3.73% / 8.70% / 5.22% | 3.51% / 2.85% / 3.14% | 3.67% / 88.45% / 7.05% |

### Immediate-Feedback Oracle

| ML model | Preprocessing | Overall flagged P / R / F1 | Approve P / R / F1 | Review P / R / F1 | Block P / R / F1 |
|---|---|---:|---:|---:|---:|
| CatBoost | R0 Control | 11.18% / 25.46% / 15.54% | 2.99% / 74.54% / 5.74% | 6.71% / 5.45% / 6.02% | 13.66% / 20.01% / 16.24% |
| CatBoost | R1 UID | 10.89% / 24.55% / 15.09% | 3.02% / 75.45% / 5.81% | 6.92% / 5.63% / 6.21% | 13.13% / 18.93% / 15.50% |
| CatBoost | R3 Full Causal + Identity | 10.41% / 23.32% / 14.39% | 3.07% / 76.68% / 5.90% | 6.36% / 5.16% / 5.70% | 12.71% / 18.16% / 14.95% |
| CatBoost | R2 Full Causal | 9.58% / 20.74% / 13.10% | 3.16% / 79.26% / 6.08% | 6.37% / 5.18% / 5.71% | 11.50% / 15.56% / 13.22% |
| XGBoost | R3 Full Causal + Identity | 9.41% / 20.39% / 12.88% | 3.18% / 79.61% / 6.11% | 6.23% / 5.06% / 5.58% | 11.31% / 15.33% / 13.02% |
| XGBoost | R2 Full Causal | 8.96% / 19.12% / 12.20% | 3.22% / 80.88% / 6.20% | 6.33% / 5.14% / 5.67% | 10.58% / 13.98% / 12.05% |
| XGBoost | R1 UID | 8.97% / 19.19% / 12.23% | 3.22% / 80.81% / 6.19% | 6.20% / 5.04% / 5.56% | 10.67% / 14.16% / 12.17% |
| XGBoost | R0 Control | 8.67% / 18.36% / 11.78% | 3.25% / 81.64% / 6.25% | 6.26% / 5.08% / 5.61% | 10.18% / 13.28% / 11.52% |
| LightGBM | R0 Control | 9.24% / 19.64% / 12.56% | 3.20% / 80.36% / 6.16% | 6.33% / 5.14% / 5.67% | 11.03% / 14.50% / 12.53% |
| LightGBM | R1 UID | 8.83% / 18.50% / 11.95% | 3.24% / 81.50% / 6.24% | 6.44% / 5.23% / 5.78% | 10.34% / 13.27% / 11.62% |
| LightGBM | R3 Full Causal + Identity | 8.91% / 18.64% / 12.06% | 3.24% / 81.36% / 6.23% | 6.50% / 5.28% / 5.83% | 10.45% / 13.36% / 11.73% |
| LightGBM | R2 Full Causal | 8.63% / 18.07% / 11.68% | 3.26% / 81.93% / 6.27% | 6.24% / 5.07% / 5.60% | 10.14% / 13.00% / 11.39% |
| Isolation Forest | R2 Full Causal | 3.83% / 6.10% / 4.70% | 3.66% / 93.90% / 7.05% | 4.15% / 3.27% / 3.66% | 3.51% / 2.82% / 3.13% |
| SGD One-Class SVM | R3 Full Causal + Identity | 3.77% / 5.86% / 4.59% | 3.67% / 94.14% / 7.06% | 4.06% / 3.12% / 3.53% | 3.48% / 2.73% / 3.06% |
| SGD One-Class SVM | R2 Full Causal | 3.77% / 5.86% / 4.59% | 3.67% / 94.14% / 7.06% | 4.06% / 3.12% / 3.53% | 3.48% / 2.73% / 3.06% |

## Historical LightGBM Feedback Sensitivity (Seeds 42–44)

Separate historical experiment using the LightGBM R1 UID scores. The 30-day blocked-label delay is a simulation assumption, not a measured real-world delay. Values are mean ± standard deviation.

| Feedback setting | Cost / Oracle Tax | Mean actions A/B/R | Feedback summary | PR-AUC |
|---|---|---|---|---:|
| Blocked labels hidden | • Cost: 2,883,768.62 ± 12,939.50 · • Tax: 1,501,533.93 ± 18,719.29 | 23,750/205,419/7,047 | • Labels received: 27,505 · • Block labels censored | 0.485435 |
| Blocked labels after 30 days | • Cost: 1,876,874.63 ± 11,928.92 · • Tax: 494,639.94 ± 17,762.71 | 150,264/78,905/7,047 | • Labels received: 153,805 · • Block labels released: 76,521 · • Pending: 2,385 | 0.485435 |
| Immediate-feedback oracle | • Cost: 1,382,234.68 ± 8,299.71 · • Tax: 0 | 218,412/10,757/7,047 | • Labels received: 236,216 · • Released after action | 0.485435 |

SOTA comparison is not directly comparable: leaderboard metrics, splits and evaluation protocols differ. Source metrics: the verified consolidated CSVs under `outputs/`, Kaggle output artifacts and `RUN_LOG.md`. Historical metrics: `artifacts/simulation/feedback_sensitivity_summary.csv` and `feedback_sensitivity_metrics.csv`.





