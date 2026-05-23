# Epileptic Seizure Recognition

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Jupyter Notebook](https://img.shields.io/badge/notebook-Phase%201%20%2B%20Phase%202-orange)](https://github.com/adhamhaithameid/epileptic-seizure-recognition)
[![Colab](https://img.shields.io/badge/Colab-Open%20Notebook-yellow)](https://colab.research.google.com/drive/1ihmUtrUv8hyeJsxGvSS-gRnxTNXCd9KF)
[![DOI](https://img.shields.io/badge/research-study-blueviolet)](https://docs.google.com/document/d/1hQM3NtDMTqfUUA7SnneqblxTEQdRf8pzPM7H7q_PGhA/edit)
[![GitHub stars](https://img.shields.io/github/stars/adhamhaithameid/epileptic-seizure-recognition)](https://github.com/adhamhaithameid/epileptic-seizure-recognition)
[![GitHub license](https://img.shields.io/github/license/adhamhaithameid/epileptic-seizure-recognition)](https://github.com/adhamhaithameid/epileptic-seizure-recognition/blob/main/LICENSE)

A two-phase soft-computing study for **epileptic seizure recognition** from EEG signals using the UCI Epileptic Seizure Recognition dataset. Phase 1 benchmarks 10 classifiers across preprocessing, reduction, and feature-selection strategies. Phase 2 introduces a **Genetic Algorithm (GA)** for evolutionary feature selection.

**Video presentation:** [Epileptic Seizure Recognition — GA Feature Selection](https://www.youtube.com/watch?v=p0bmKpUwvBY)

---

## Table of Contents

- [About](#about)
- [Dataset](#dataset)
- [Phase 1 — Model Benchmarking](#phase-1--model-benchmarking)
- [Phase 2 — GA Feature Selection](#phase-2--ga-feature-selection)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Resources](#resources)
- [License](#license)
- [Citation](#citation)

---

## About

**Course:** CSC425 — Math for Data Science (Soft Computing)

This project evaluates machine-learning methods for detecting epileptic seizures from EEG-derived features. It is structured in two phases:

| Phase | Focus | Key Methods |
|-------|-------|-------------|
| **1** | Full benchmarking pipeline | 10 classifiers, 5 reduction techniques, filter/wrapper/embedded selection |
| **2** | Evolutionary feature selection | Genetic Algorithm vs. RFE / SelectKBest / Embedded RF |

**Best result (Phase 1):** SVM (RBF) — **97.52% test accuracy**, **ROC AUC = 0.9972**

---

## Dataset

- **Source:** [UCI Epileptic Seizure Recognition Dataset](https://archive.ics.uci.edu/dataset/213/epileptic+seizure+recognition)
- **Samples:** 11,500
- **Features:** 178 EEG-point attributes (`X1`–`X178`)
- **Target:** Binary — seizure (class 1) vs. non-seizure (classes 2–5)
- **Split:** 80/20 stratified train/test, 5-fold cross-validation

---

## Phase 1 — Model Benchmarking

Phase 1 builds a complete end-to-end pipeline:

1. **Loading & cleaning** — missing-value handling, numeric coercion
2. **Statistical analysis** — Chi-square, t-test, ANOVA, descriptive stats
3. **Feature reduction** — PCA, Kernel PCA, LDA, SVD
4. **Feature selection** — SelectKBest, RFE, Embedded RF importance
5. **Model training & evaluation** — 10 classifiers with 5-fold CV

### Classifiers Evaluated

| Model | Test Accuracy | F1 Score | ROC AUC |
|-------|:------------:|:--------:|:-------:|
| SVM (RBF) | **0.9752** | **0.9363** | **0.9972** |
| Feed Forward Neural Network | 0.9683 | 0.9200 | 0.9901 |
| Naive Bayesian | 0.9561 | 0.8906 | 0.9838 |
| Feed Back Neural Network | 0.9504 | 0.8665 | 0.9751 |
| Decision Tree (Entropy) | 0.9409 | 0.8400 | 0.9201 |
| KNN (Euclidean) | 0.9313 | 0.7937 | 0.9207 |
| KNN (Manhattan) | 0.9261 | 0.7739 | 0.9212 |
| Bayesian Belief Network | 0.8448 | 0.6925 | 0.9309 |
| Logistic Regression | 0.8165 | 0.1594 | 0.4992 |
| LDA Classifier | 0.8122 | 0.1184 | 0.4936 |

---

## Phase 2 — GA Feature Selection

Phase 2 implements a **binary Genetic Algorithm** that evolves optimal feature subsets.

### GA Configuration

| Parameter | Value |
|-----------|-------|
| Population | 30 |
| Generations | 20 |
| Crossover prob. | 0.80 |
| Mutation prob. | 0.01 (per gene) |
| Selection | Tournament (k=3) |
| Elitism | 2 |
| Min. features | 5 |
| Fitness | 5-fold CV accuracy — λ · (selected / total) |

### Comparison Baselines

- RFE (Logistic Regression estimator)
- SelectKBest (ANOVA)
- Embedded RF importance
- All-features baseline

**Evaluation classifiers:** SVM (RBF), Decision Tree

**Planned outputs:** evolution curves, accuracy/F1 bar charts, Jaccard overlap heatmap, ROC curves.

> Note: Phase 2 code is complete and documented. Results need one full execution pass to persist output cells.

---

## Results Summary

| Metric | Phase 1 Best (SVM RBF) |
|--------|:---------------------:|
| Test Accuracy | **97.52%** |
| ROC AUC | **0.9972** |
| F1 Score | 0.9363 |
| Precision | 0.9443 |
| Recall | 0.9284 |

Non-linear models (SVM, neural networks) significantly outperform linear baselines, consistent with the complex, non-linear nature of EEG signals.

---

## Repository Structure

```
.
├── Phase 1 Colab.ipynb           # Phase 1 notebook (full saved outputs)
├── phase 2 colab.ipynb           # Phase 2 GA notebook (code complete)
├── Full_Documentation.md         # Full project documentation
├── Full_Documentation.docx       # Word version of documentation
├── Research_Study.md             # Formal research study draft
├── Research_Study.docx           # Word version of research study
├── README.md                     # This file
├── Soft-Computing-Research-Study.pptx  # Presentation deck
├── CSC425-Math for Data Science-Project Phase 1-tasks Cover Sheet.docx
├── CSC425-Math for Data Science-Project Phase 2-tasks Cover Sheet.docx
├── Project Information Template.docx
└── outputs/                      # Presentation build assets
    └── 019e5090-a227-75c0-bd53-54cdbae187fa/
        └── presentations/epileptic-seizure-study/
```

### Branches

| Branch | Description |
|--------|-------------|
| `main` | Deliverables: notebooks, documentation, presentation |
| `experimental/new-stuff-migration` | Full Phase 1 Python script + GA notebook + raw CSV |
| `experimental/simpler-epileptic-version` | Cleaned, simplified Phase 1 pipeline |

---

## Quick Start

```bash
# Clone the repo
git clone https://github.com/adhamhaithameid/epileptic-seizure-recognition.git
cd epileptic-seizure-recognition

# Open the notebooks
jupyter notebook "Phase 1 Colab.ipynb"
```

### Run in Google Colab

| Phase | Link |
|-------|------|
| Phase 1 | [Open in Colab](https://colab.research.google.com/drive/1ihmUtrUv8hyeJsxGvSS-gRnxTNXCd9KF) |
| Phase 2 | [Open in Colab](https://colab.research.google.com/drive/1b0rBmBzEgozOo8VIo749H0kYiyZjX8ar) |

### Dependencies

- Python 3.11+
- numpy, pandas, scikit-learn, matplotlib, seaborn, scipy

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy
```

---

## Resources

| Resource | Link |
|----------|------|
| Video presentation | [YouTube](https://www.youtube.com/watch?v=p0bmKpUwvBY) |
| Phase 1 Colab notebook | [Colab](https://colab.research.google.com/drive/1ihmUtrUv8hyeJsxGvSS-gRnxTNXCd9KF) |
| Phase 2 Colab notebook | [Colab](https://colab.research.google.com/drive/1b0rBmBzEgozOo8VIo749H0kYiyZjX8ar) |
| Research study (Google Doc) | [Doc](https://docs.google.com/document/d/1hQM3NtDMTqfUUA7SnneqblxTEQdRf8pzPM7H7q_PGhA/edit) |
| Full documentation (Google Doc) | [Doc](https://docs.google.com/document/d/1uv1zV8gsoTtub3P19x0MYCM4oLX3UCgt2Hi6ZSQGMzM/edit) |
| GitHub repo | [GitHub](https://github.com/adhamhaithameid/epileptic-seizure-recognition) |

---

## License

This project is for educational purposes as part of CSC425 — Math for Data Science at the University of Jeddah.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{epileptic-seizure-recognition,
  author = {Adham Haitham Eid},
  title = {Epileptic Seizure Recognition: A Two-Phase Soft Computing Study},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/adhamhaithameid/epileptic-seizure-recognition}
}
```
