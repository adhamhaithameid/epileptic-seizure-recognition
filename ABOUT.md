# About This Project

## Overview

This repository contains a **two-phase soft-computing research study** on **epileptic seizure recognition** using the UCI Epileptic Seizure Recognition dataset. The project was developed for **CSC425 — Math for Data Science** at the **University of Jeddah**.

The core problem: given 178 EEG-derived features per sample, can we reliably detect whether a patient is experiencing an epileptic seizure? The study tackles this through systematic model benchmarking (Phase 1) and evolutionary feature optimization (Phase 2).

## Phase 1: Full Benchmarking Pipeline

Phase 1 implements a complete end-to-end machine-learning workflow:

- **Preprocessing:** data cleaning, missing-value handling, descriptive statistics
- **Statistical testing:** Chi-square, t-test, ANOVA for feature-target association
- **Dimensionality reduction:** PCA, Kernel PCA, LDA projection, SVD
- **Feature selection:** SelectKBest (filter), RFE (wrapper), Embedded Random Forest (embedded)
- **Classifiers (10):** Naive Bayesian, Bayesian Belief Network, Decision Tree, LDA, Feed-Forward NN, Feedback NN, KNN (Euclidean & Manhattan), SVM (RBF), Logistic Regression
- **Evaluation:** 5-fold stratified cross-validation, accuracy, F1, ROC AUC, confusion matrices, learning curves

**Key finding:** Non-linear models dramatically outperform linear ones. SVM (RBF) achieves **97.52% accuracy** with **ROC AUC = 0.9972**, confirming the complex non-linear nature of EEG seizure patterns.

## Phase 2: Genetic Algorithm Feature Selection

Phase 2 extends the work with an evolutionary approach to feature selection:

- Binary chromosome encoding (1 = feature selected)
- Tournament selection, single-point crossover, bit-flip mutation
- Elitism to preserve top solutions
- Fitness: 5-fold CV accuracy with regularization penalty
- Compared against: RFE, SelectKBest, Embedded RF, All-Features baseline
- Evaluated on: SVM (RBF) and Decision Tree classifiers

The GA approach is methodologically significant because it searches the feature-subset space globally rather than greedily, potentially discovering non-obvious feature interactions that maintain or improve performance with fewer dimensions.

## Results

| Model | Test Accuracy | ROC AUC |
|-------|:------------:|:-------:|
| SVM (RBF) | **97.52%** | **0.9972** |
| Feed Forward NN | 96.83% | 0.9901 |
| Naive Bayesian | 95.61% | 0.9838 |
| Decision Tree | 94.09% | 0.9201 |

## Significance

This work demonstrates that combining **soft computing** (evolutionary algorithms) with **classical machine learning** yields a powerful framework for medical EEG classification. The systematic benchmarking provides a reproducible baseline, and the GA framework offers a path toward interpretable, low-dimensional feature subsets for clinical deployment.

## Branches

- **`main`** — Deliverables: notebooks with saved outputs, documentation, research study, presentation
- **`experimental/new-stuff-migration`** — Full Phase 1 Python script, GA notebook, raw CSV data
- **`experimental/simpler-epileptic-version`** — Cleaned, simplified Phase 1 pipeline

## Authors

- **Adham Haitham Eid** — University of Jeddah, CSC425

## Links

- GitHub: https://github.com/adhamhaithameid/epileptic-seizure-recognition
- Phase 1 Colab: https://colab.research.google.com/drive/1ihmUtrUv8hyeJsxGvSS-gRnxTNXCd9KF
- Phase 2 Colab: https://colab.research.google.com/drive/1b0rBmBzEgozOo8VIo749H0kYiyZjX8ar
- Video: https://www.youtube.com/watch?v=p0bmKpUwvBY
- Research Study: https://docs.google.com/document/d/1hQM3NtDMTqfUUA7SnneqblxTEQdRf8pzPM7H7q_PGhA/edit
- Documentation: https://docs.google.com/document/d/1uv1zV8gsoTtub3P19x0MYCM4oLX3UCgt2Hi6ZSQGMzM/edit
