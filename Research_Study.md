# Research Study

## Title

Soft Computing for Epileptic Seizure Recognition: A Two-Phase Study Combining Model Benchmarking and Genetic Algorithm Feature Selection

## Abstract

This study investigates epileptic seizure recognition using soft computing and machine learning techniques on the UCI Epileptic Seizure Recognition dataset. In Phase 1, a full benchmarking pipeline was implemented, including preprocessing, statistical analysis, dimensionality reduction, and multi-model evaluation. The strongest model achieved 97.52% test accuracy with ROC AUC 0.9972 (SVM with RBF kernel). In Phase 2, we designed a Genetic Algorithm (GA) feature-selection framework to reduce feature dimensionality while maintaining classifier performance, and to compare GA-selected subsets against classical methods (RFE, SelectKBest, embedded importance). At documentation time, Phase 2 code is complete but saved execution outputs are not present in the notebook file; therefore, this study reports complete Phase 1 quantitative results and a full Phase 2 protocol ready for final experimental completion.

## 1. Introduction

Automatic seizure detection from EEG-derived signals can support clinical monitoring and improve decision-making speed. However, practical challenges include high-dimensional feature spaces, class imbalance, and nonlinear signal behavior. This work addresses those challenges through a structured two-phase approach:

- Phase 1: robust baseline benchmarking across diverse classifiers.
- Phase 2: GA-based feature subset optimization and comparative validation.

## 2. Materials and Methods

### 2.1 Dataset

The experiments use the UCI epileptic seizure recognition dataset structure with 11,500 samples and 178 EEG features per sample. Labels were transformed to a binary seizure vs non-seizure target for the core classification task.

### 2.2 Phase 1 Methodology

Phase 1 includes:

- Data loading with fallback download support
- Cleaning and numeric coercion
- Train/test split (80/20, stratified)
- 5-fold cross-validation
- Statistical analysis and visualization
- Feature reduction/selection techniques (PCA, Kernel PCA, LDA, SVD, SelectKBest, RFE, embedded RF)
- Comparative model evaluation

Classifiers evaluated:

- Naive Bayesian
- Bayesian Belief Network
- Decision Tree (Entropy)
- LDA Classifier
- Feed Forward Neural Network
- Feedback Neural Network
- KNN (Euclidean, Manhattan)
- SVM (RBF)
- Logistic Regression

Evaluation metrics:

- Test Accuracy
- F1 Score
- ROC AUC
- Error Rate
- Confusion Matrix
- CV Accuracy

### 2.3 Phase 2 Methodology (GA Feature Selection)

Phase 2 implements a binary Genetic Algorithm where each chromosome encodes selected features.

GA configuration:

- Population = 30
- Generations = 20
- Crossover probability = 0.80
- Mutation probability = 0.01
- Tournament selection (k = 3)
- Elitism = 2
- Minimum selected features = 5
- Fitness = 5-fold CV accuracy - lambda * (selected/total)

Baseline feature methods:

- RFE (Logistic Regression estimator)
- SelectKBest (ANOVA)
- Embedded RF feature importance
- All-features baseline

Evaluation classifiers in Phase 2:

- SVM (RBF)
- Decision Tree

## 3. Results

### 3.1 Phase 1 Quantitative Results

| Model | Test Accuracy | F1 Score | ROC AUC |
|---|---:|---:|---:|
| SVM (RBF Kernel) | 0.9752 | 0.9363 | 0.9972 |
| Feed Forward Neural Network | 0.9683 | 0.9200 | 0.9901 |
| Naive Bayesian | 0.9561 | 0.8906 | 0.9838 |
| Feed Back Neural Network | 0.9504 | 0.8665 | 0.9751 |
| Decision Tree (Entropy) | 0.9409 | 0.8400 | 0.9201 |
| KNN (Euclidean) | 0.9313 | 0.7937 | 0.9207 |
| KNN (Manhattan) | 0.9261 | 0.7739 | 0.9212 |
| Bayesian Belief Network | 0.8448 | 0.6925 | 0.9309 |
| Logistic Regression | 0.8165 | 0.1594 | 0.4992 |
| LDA Classifier | 0.8122 | 0.1184 | 0.4936 |

Best Phase 1 result:

- SVM (RBF), test accuracy 0.9752, ROC AUC 0.9972.

### 3.2 Phase 2 Current Status

The Phase 2 notebook provides full implementation and planned visual/metric outputs. However, there are currently no saved output cells in `phase 2 colab.ipynb`. Therefore, this draft intentionally avoids inventing Phase 2 numeric values and treats them as pending reproducible execution.

## 4. Discussion

Phase 1 results indicate that non-linear models (SVM RBF and neural networks) are substantially better suited for this dataset than linear baselines. This is consistent with EEG signal complexity and potential nonlinear class boundaries.

The GA extension is methodologically meaningful because it optimizes feature subsets globally rather than greedily. Compared to RFE and simple filters, GA can capture interactions among features and may preserve performance with fewer dimensions. The tradeoff is computation time due to repeated CV scoring during evolution.

## 5. Limitations

- Phase 2 final quantitative outputs are not yet persisted in notebook outputs.
- Potential sensitivity to random seed and split configuration.
- Accuracy must be interpreted with class imbalance awareness; F1 and ROC AUC remain essential.

## 6. Conclusion

This two-phase study establishes a strong seizure-recognition baseline and a clear pathway to feature-optimized modeling. Phase 1 confirms high performance of SVM and neural models. Phase 2 introduces a complete, reproducible GA framework for feature selection and comparative evaluation against established methods. The next step is a full Phase 2 execution pass and insertion of final GA-vs-baseline results into the final manuscript.

## 7. Future Work

- Execute Phase 2 end-to-end and save outputs
- Add confidence intervals over repeated splits
- Evaluate additional imbalance handling strategies
- Test multi-objective GA (accuracy vs subset size Pareto front)

## References

- UCI Epileptic Seizure Recognition Dataset
- scikit-learn official documentation
- Reference papers listed in the Phase 1 notebook related-work table

