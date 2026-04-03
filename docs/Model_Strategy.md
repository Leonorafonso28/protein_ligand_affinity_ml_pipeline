# Model Strategy

## Overview

This document describes the machine learning modelling decisions made in this project, covering 
label definition, model architectures, hyperparameter optimisation, evaluation design, and 
key results. Two parallel modelling branches are maintained throughout — one for **pKi** and 
one for **pChEMBL** — enabling direct comparison of label source impact on model performance.

---

## 1. Problem Framing

### Task

**Binary classification** of protein–ligand binding affinity. Each sample is a unique 
protein chain–ligand pair from an experimentally validated PDB co-complex, enriched with 
ChEMBL bioactivity data.

### Label Binarisation

Both pKi and pChEMBL values are converted to binary labels using a threshold of **6.5**:

| Label | Condition | Approximate potency |
|---|---|---|
| Active (1) | value ≥ 6.5 | Ki / IC50 ≤ ~316 nM |
| Inactive (0) | value < 6.5 | weaker binders |

A value of 6.5 on the −log₁₀ molar scale is a widely accepted industry convention for 
primary screening hit cutoffs in ChEMBL-based activity classification benchmarks.

### Label Branches

| Branch | Source | Coverage | Homogeneity |
|---|---|---|---|
| **pKi** | Equilibrium inhibition constant (Ki only) | Smaller dataset | High — single assay type |
| **pChEMBL** | ChEMBL-standardised multi-assay affinity | Larger dataset | Lower — Ki, Kd, IC50, EC50 combined |

The pKi branch is preferred when assay-type consistency is required (e.g. free energy 
perturbation benchmarking). The pChEMBL branch provides broader target coverage and is 
preferred for general-purpose models and multi-task learning.

---

## 2. Model Architectures

### 2.1 XGBoost Classifier

An **XGBoost binary classifier** (`objective='binary:logistic'`) was trained on the 
tabular feature matrix. XGBoost was chosen for its strong performance on structured/tabular 
data, native handling of feature interactions, built-in regularisation, and interpretability 
via feature importance scores.

**Fixed parameters:**
- `objective`: `binary:logistic`
- `eval_metric`: `logloss`
- `random_state`: 42

**Tuned parameters** (via Optuna, see Section 3):

| Parameter | Description | Search Range |
|---|---|---|
| `lambda` | L2 regularisation | log-uniform [1e-5, 1.0] |
| `alpha` | L1 regularisation | log-uniform [1e-5, 1.0] |
| `colsample_bytree` | Features sampled per tree | uniform [0.5, 1.0] |
| `subsample` | Rows sampled per tree | uniform [0.5, 1.0] |
| `learning_rate` | Step size shrinkage | uniform [0.01, 0.1] |
| `max_depth` | Maximum tree depth | int [3, 10] |
| `min_child_weight` | Minimum leaf instance weight | int [1, 10] |
| `n_estimators` | Number of boosting rounds | int [100, 1000] |

---

### 2.2 Deep Neural Network (DNN)

A fully-connected **feed-forward DNN** was implemented in TensorFlow/Keras as a 
complementary architecture. DNNs were chosen to assess whether learned non-linear 
transformations of the feature space could outperform the gradient boosting approach.

**Architecture search space** (via Optuna):

| Hyperparameter | Range |
|---|---|
| Number of hidden layers | 1 – 5 |
| Units per layer | 16 – 128 |
| Activation function | ReLU, tanh, ELU |
| Dropout rate (per layer) | 0.0 – 0.5 |
| L2 weight decay | log-uniform [1e-6, 1e-2] |
| Learning rate (Adam) | log-uniform [1e-5, 1e-2] |
| Batch size | 16, 32, 64 |
| Epochs | 50 – 150 |

**Regularisation:** L2 weight decay and per-layer dropout are tuned simultaneously, 
allowing the search to find the optimal bias–variance trade-off for the high-dimensional 
input space (~1,490 features).

**Early stopping:** patience = 10 epochs on `val_loss`, with best-weight restoration — 
prevents overfitting during extended training runs.

**Reproducibility:** Python, NumPy, TensorFlow, and OS hash seeds are all fixed at 42. 
Threading controls (`OMP_NUM_THREADS=1`) are set in evaluation scripts for 
bit-reproducible inference.

---

## 3. Hyperparameter Optimisation

Both architectures use **Optuna** with the **Tree-structured Parzen Estimator (TPE)** 
sampler for Bayesian hyperparameter search.

| Setting | Value |
|---|---|
| Sampler | TPESampler (seed = 42) |
| Objective | Maximise validation **F1-score** |
| Trials | 400 |
| Execution order | HPO script first → evaluation script |

**Why F1?** The pKi and pChEMBL datasets are class-imbalanced (more actives than 
inactives). F1 balances precision and recall, enabling the model to learn from the minority 
class without significantly increasing the false positive rate — a critical property for 
virtual screening applications.

Best hyperparameter configurations are persisted to JSON files and loaded by the 
corresponding evaluation scripts, ensuring full reproducibility of the final model.

---

## 4. Evaluation Design

### 4.1 Six-Partition Splitting Strategy

A rigorous blind splitting strategy was implemented to evaluate generalisation across three 
orthogonal axes, directly mapping to real pharmaceutical deployment scenarios:

| Partition | Construction | Generalisation axis |
|---|---|---|
| **Train** (~72% of total) | Remaining data, 80% split | In-distribution learning |
| **Validation** (~8%) | Remaining data, 10% split | HPO and early stopping |
| **Test** (~8%) | Remaining data, 10% split | In-distribution generalisation |
| **Blind** | 15% SMILES pool ∩ 15% sequence pool | Both protein AND ligand unseen |
| **Blind Protein** | 15% sequence pool (excl. Blind) | Novel targets — target discovery |
| **Blind Ligand** | 15% SMILES pool (excl. Blind) | Novel scaffolds — virtual screening |

All splits are performed over **unique SMILES** and **unique sequences**, not at the row 
level, preventing any form of data leakage. Seed fixed at 42 for reproducibility.

### 4.2 Evaluation Metrics

All partitions are evaluated using the following metrics at a decision threshold of 0.5:

| Metric | Rationale |
|---|---|
| **AUC-ROC** | Threshold-independent ranking ability; primary discrimination metric |
| **MCC** | Best single metric for imbalanced binary classification; accounts for all four confusion matrix cells |
| **F1-Score** | Harmonic mean of precision and recall; the HPO optimisation objective |
| **Recall (Sensitivity)** | Fraction of true actives identified; critical for virtual screening (missing actives is costly) |
| **Precision** | Fraction of predicted actives that are truly active; important for candidate prioritisation |
| **Accuracy** | Overall correctness; interpreted alongside MCC given class imbalance |

---

## 5. Results Summary

### pKi Branch — XGBoost

| Dataset | AUC | Accuracy | Recall | Precision | F1 | MCC |
|---|---|---|---|---|---|---|
| Test set | 0.9362 | 0.9236 | 0.9502 | 0.9455 | 0.9479 | 0.8051 |
| Blind set | 0.8351 | 0.8293 | 0.9118 | 0.8857 | 0.8986 | 0.3623 |
| Blind Protein | 0.9677 | 0.9425 | 0.9815 | 0.9438 | 0.9623 | 0.8446 |
| Blind Ligand | 0.8480 | 0.8447 | 0.9178 | 0.8872 | 0.9022 | 0.5276 |

### pChEMBL Branch — XGBoost

| Dataset | AUC | Accuracy | Recall | Precision | F1 | MCC |
|---|---|---|---|---|---|---|
| Test set | 0.9573 | 0.8984 | 0.9291 | 0.9225 | 0.9258 | 0.7646 |
| Blind set | 0.8583 | 0.7949 | 0.9050 | 0.8009 | 0.8498 | 0.5402 |
| Blind Protein | 0.9497 | 0.8962 | 0.9453 | 0.9029 | 0.9236 | 0.7640 |
| Blind Ligand | 0.8567 | 0.8184 | 0.8981 | 0.8379 | 0.8669 | 0.5855 |

### Key Findings

- **XGBoost outperforms DNN** consistently across both label branches and all evaluation 
  partitions, particularly on the metrics most relevant for imbalanced datasets 
  (MCC, F1, precision, recall).
- **pKi models achieve slightly higher metrics** than pChEMBL models, consistent with the 
  hypothesis that pChEMBL's aggregation of multiple assay types introduces label noise.
- **Blind Protein performance is high**, approaching test set levels — suggesting that 
  ESM-2 embeddings effectively capture cross-target generalisable protein features. 
  Note: the absence of sequence identity filtering may partially inflate these results.
- **Blind Ligand performance is lower**, indicating t