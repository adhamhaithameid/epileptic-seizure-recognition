# Project Title
Epileptic Seizure Recognition with a Full Cartesian Soft Computing Benchmark

## Authors
- Student Name: [Fill Your Name]
- Course: Soft Computing (CSC425)
- Instructor: Dr. Ahmed Anter
- Semester: Spring 2026

## Abstract
This project presents an end-to-end soft computing workflow for epileptic seizure recognition using a fully refactored and reproducible Cartesian benchmark pipeline. The workflow covers all required stages: preprocessing, feature reduction, feature selection, classification, evaluation, visualization, and paper-ready reporting.

Two prediction tracks are evaluated: binary seizure detection and multiclass classification. The benchmark combines four preprocessing methods, four reduction options, eight selection strategies, and six classifiers under three-fold stratified cross-validation. This yields 1,536 unique method combinations and 4,608 fold-level evaluations.

Invalid combinations are handled safely through auto-fix and skip logging (`status`, `skip_reason`) so execution continues and coverage remains auditable. Final outputs include per-combination metrics, ranking tables, baseline deltas, comparison reports, and figure suites (heatmaps, top-N bars, fold-variance, ROC for top binary pipelines).

## Keywords
Soft Computing, Epileptic Seizure Recognition, Cartesian Benchmark, Feature Reduction, Feature Selection, Genetic Algorithm, Classification

## 1. Introduction
Epileptic seizure recognition is a high-impact classification problem where robust machine learning can support decision making in neurological analysis. A key challenge in course projects is not only obtaining good accuracy, but proving fair and reproducible comparisons across many techniques.

To address this, the project was redesigned as a Cartesian benchmark framework where each method family is explicitly enumerated and evaluated under a common protocol. This allows direct, transparent comparison between preprocessing, reduction, selection, and model choices.

## 2. Related Work
Add at least 10 seizure/EEG classification studies with method and metric comparisons.

| Ref No. | Paper | Year | Methods | Reported Results |
|:--:|:--|:--:|:--|:--|
| [R1] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R2] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R3] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R4] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R5] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R6] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R7] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R8] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R9] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |
| [R10] | [Add paper] | [Year] | [Methods] | [Accuracy/F1/AUC] |

## 3. Methodology
### 3.1 Dataset
- Dataset: Epileptic Seizure Recognition
- Samples: 11,500
- Features: 178 numeric predictors
- Targets:
  - Binary track: class 1 vs classes 2-5
  - Multiclass track: original five classes

### 3.2 Preprocessing
- `standard`
- `minmax`
- `robust`
- `quantile`

### 3.3 Feature Reduction
- `none`
- `pca`
- `lda_projection`
- `svd`

### 3.4 Feature Selection
- `none`
- `filter_chi2`
- `filter_anova`
- `filter_correlation`
- `wrapper_sfs`
- `wrapper_rfe`
- `embedded_l1`
- `ga_selection`

### 3.5 Classifiers
- `knn`
- `svm`
- `decision_tree`
- `logistic_regression`
- `lda_classifier`
- `mlp_ann`

### 3.6 Evaluation Protocol
- 3-fold stratified CV
- Metrics: accuracy, precision, recall, f1, roc_auc (binary), error_rate
- Runtime metrics: fit and prediction time
- Failure handling: logged per fold with `status` and `skip_reason`

## 4. Proposed Model
The project uses a deterministic staged Cartesian engine:
1. Load and clean data.
2. For each track and fold, apply each preprocessing method.
3. Apply each reduction method.
4. Apply each selection method.
5. Train/evaluate each classifier.
6. Save fold-level metrics and status rows.
7. Aggregate rankings, baseline deltas, and plots.

Combination math:
- Unique combos = `4 x 4 x 8 x 6 x 2 = 1536`
- Fold evaluations = `1536 x 3 = 4608`

## 5. Results and Discussion
### 5.1 Required result files
- `results/metrics/cartesian_metrics_all.csv`
- `results/metrics/cartesian_run_manifest.json`
- `results/tables/cartesian_summary_by_combo.csv`
- `results/tables/cartesian_rankings_binary.csv`
- `results/tables/cartesian_rankings_multiclass.csv`
- `results/reports/cartesian_comparison_report.md`

### 5.2 Result insertion points
After full run completion, paste:
- Top 10 binary rows from `cartesian_rankings_binary.csv`
- Top 10 multiclass rows from `cartesian_rankings_multiclass.csv`
- Skip/failure summary from `cartesian_metrics_all.csv` (`status`, `skip_reason`)
- Best-model interpretation from `cartesian_comparison_report.md`

### 5.3 Discussion prompts
- Which preprocessing method appears most stable by fold variance?
- Which reduction/selection families improve F1 vs baseline (`none + none`)?
- Which techniques fail/skip most frequently and why?

## 6. Conclusion and Future Work
This project delivers a complete, reproducible soft-computing benchmark that evaluates all requested method families in one framework. The refactor improves traceability, output organization, and experimental rigor by enforcing full Cartesian accounting and standardized schemas.

Future work:
1. Add hyperparameter optimization for top-ranked combinations.
2. Expand to additional datasets from the course list.
3. Add statistical significance tests between top pipelines.

## 7. References
Use APA style and map citations consistently between text and bibliography.
