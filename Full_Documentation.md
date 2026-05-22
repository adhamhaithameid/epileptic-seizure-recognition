# Full Project Documentation

## 1) Project Overview

This project studies epileptic seizure recognition using soft computing and machine learning methods on the UCI Epileptic Seizure Recognition dataset. It is implemented in two phases:

- Phase 1 notebook: `Phase 1 Colab.ipynb`
- Phase 2 notebook: `phase 2 colab.ipynb`

The goal is to build a reproducible end-to-end workflow for preprocessing, feature engineering, model comparison, and advanced feature selection using a Genetic Algorithm (GA).

## 2) Data and Problem Setup

### Dataset

- Source family: UCI Epileptic Seizure Recognition
- Samples: 11,500
- Feature columns: 178 EEG-point attributes (`X1..X178`)
- Label: original multiclass label transformed into binary target (seizure vs non-seizure)

### Data Split and Validation

- Train/test split: 80/20 (stratified)
- Cross-validation: 5-fold stratified CV
- Fixed random state: 42

## 3) Phase 1 Technical Workflow

Phase 1 builds the foundational benchmarking pipeline.

### 3.1 Setup and Utilities

- Imports and global plotting configuration
- Custom Bayesian classifier class
- Regression metric helper functions (MAE, RMSE, R2, Willmott d, NSE, Legates-McCabe)

### 3.2 Dataset Loading and Target Construction

- Reads dataset from local path with fallback download logic
- Drops unnamed columns if present
- Converts predictors to numeric format
- Converts original class labels into binary target

### 3.3 Preprocessing and Analysis

- Missing-value handling and cleaning
- Descriptive statistics
- Statistical feature tests:
  - Chi-square
  - t-test
  - ANOVA
- Visualizations and rubric-aligned exploratory outputs

### 3.4 Feature Reduction and Selection

Implemented methods:

- PCA
- Kernel PCA
- LDA projection
- SVD
- SelectKBest
- RFE
- Embedded feature importance (Random Forest)

### 3.5 Model Training and Evaluation

Evaluated classifiers include:

- Naive Bayesian
- Bayesian Belief Network
- Decision Tree (Entropy)
- LDA Classifier
- Neural Network (Feed Forward)
- Feed Back Neural Network
- K-NN (Euclidean)
- K-NN (Manhattan)
- SVM (RBF Kernel)
- Logistic Regression

Metrics recorded:

- CV Accuracy
- Test Accuracy
- Error Rate
- F1 Score
- Confusion Matrix
- ROC AUC
- Fit interpretation (balanced/overfit/underfit comment)

## 4) Phase 1 Final Results (from saved notebook output)

| Model | Test Accuracy | F1 Score | ROC AUC |
|---|---:|---:|---:|
| SVM (RBF Kernel) | 0.9752 | 0.9363 | 0.9972 |
| Neural Network (Feed Forward) | 0.9683 | 0.9200 | 0.9901 |
| Naive Bayesian | 0.9561 | 0.8906 | 0.9838 |
| Feed Back Neural Network | 0.9504 | 0.8665 | 0.9751 |
| Decision Tree (Entropy) | 0.9409 | 0.8400 | 0.9201 |
| K-NN (Euclidean) | 0.9313 | 0.7937 | 0.9207 |
| K-NN (Manhattan) | 0.9261 | 0.7739 | 0.9212 |
| Bayesian Belief Network | 0.8448 | 0.6925 | 0.9309 |
| Logistic Regression | 0.8165 | 0.1594 | 0.4992 |
| LDA Classifier | 0.8122 | 0.1184 | 0.4936 |

Best-performing model in this run:

- SVM (RBF Kernel), Accuracy = 97.52%, ROC AUC = 0.9972

### Regression side metrics reported in Phase 1

- MAE: 0.31095
- RMSE: 0.41469
- R2: -0.07481
- Willmott d: 0.27945
- NSE: -0.07481
- Legates-McCabe: 0.02828

## 5) Phase 2 Technical Workflow (GA Extension)

Phase 2 extends Phase 1 by introducing feature selection with a Genetic Algorithm.

### 5.1 Key Design Choices

- Chromosome: binary mask over features (1 = selected)
- Population size: 30
- Generations: 20
- Crossover probability: 0.80
- Mutation probability: 0.01 per gene
- Tournament selection (k=3)
- Elitism: top 2 individuals
- Minimum selected features: 5
- Fitness: 5-fold CV accuracy - lambda penalty * (selected/total)

### 5.2 Comparison Protocol

Feature methods compared:

- GA (proposed)
- RFE
- SelectKBest
- Embedded RF importance
- All features baseline

Evaluation classifiers:

- SVM (RBF)
- Decision Tree

Planned outputs:

- GA evolution curves (fitness and selected feature count)
- Accuracy/F1 comparisons
- Jaccard overlap heatmap among selected subsets
- ROC curves per feature-selection method

## 6) Current Execution Status

At the time of this documentation pass:

- `Phase 1 Colab.ipynb` includes saved run outputs and final metrics table.
- `phase 2 colab.ipynb` currently has **no saved execution outputs** in the file metadata.

This means Phase 2 methodology is fully documented, but final Phase 2 quantitative outcomes still need one full execution and result capture.

## 7) Reproducibility Checklist

1. Ensure both notebooks are in the same runtime directory.
2. Run all definition cells in Phase 1, then execute the full pipeline cell.
3. Verify final model comparison table is produced.
4. Run Phase 2 setup and import utilities from Phase 1.
5. Execute GA selection and baseline methods.
6. Save notebook outputs so final tables/charts persist.

## 8) Limitations and Risk Notes

- Class imbalance may bias plain accuracy unless monitored with F1 and ROC AUC.
- GA runtime is relatively high due to repeated CV fitness evaluations.
- Results can vary if random seeds or split strategy change.
- Linear regression metrics are weak for this classification-dominant task.

## 9) Deliverables Produced in This Session

- Presentation deck (`.pptx`): integrated Phase 1 + Phase 2 story
- Full documentation file (this document)
- Formal research study draft (see `Research_Study.md`)

