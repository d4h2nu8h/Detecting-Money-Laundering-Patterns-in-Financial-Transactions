# Detecting Money Laundering Patterns in Financial Transactions

> A binary classification pipeline that identifies suspicious financial transactions indicative of money laundering, trained on 4.5 million IBM-simulated transaction records using XGBoost with undersampling to address a ~1000:1 class imbalance.

---

## Overview

Money laundering undermines the integrity of global financial systems and enables a wide range of illicit activity — from corruption and fraud to terrorism financing and organised crime. Detecting it automatically is a hard problem: laundering transactions are rare, sophisticated, and deliberately designed to blend in with legitimate activity.

This project builds a supervised binary classification system on the IBM Transactions for Anti-Money Laundering dataset, framing the problem as anomaly detection within an extreme class imbalance setting. The full pipeline covers data preprocessing, exploratory analysis of feature correlations (including Timestamp and Payment Format), undersampling to create a balanced training set, dimensionality reduction via PCA, t-SNE, and TruncatedSVD for visual analysis, and benchmarking of five classifiers with XGBoost selected as the final model via GridSearchCV hyperparameter tuning.

---

## Dataset

**Source:** IBM Transactions for Anti-Money Laundering (AML) — Kaggle

**File:** `HI-Small_Trans.csv`

| Property | Detail |
|---|---|
| Rows | 4,507,864 transactions |
| Columns | 11 features + 1 target |
| Target variable | `Is_Laundering` (binary: 0 = legitimate, 1 = suspicious) |
| Class balance | ~99.905% legitimate, ~0.095% suspicious |
| Imbalance ratio | ~1,000:1 |

**Features:**

| Feature | Type | Notes |
|---|---|---|
| Timestamp | Datetime | Converted to Unix timestamp; positively correlated with laundering labels |
| From Bank | Categorical | Label encoded |
| From Account | Categorical | Label encoded |
| To Bank | Categorical | Label encoded |
| To Account | Categorical | Label encoded |
| Amount Received | Numerical | MinMax normalised |
| Receiving Currency | Categorical | Label encoded |
| Amount Paid | Numerical | MinMax normalised |
| Payment Currency | Categorical | Label encoded |
| Payment Format | Ordinal | Ordinally encoded by complexity: Cash(1) → Cheque(2) → ACH(3) → Credit Card(4) → Wire(5) → Bitcoin(6) → Reinvestment(7); negatively correlated with laundering labels |

---

## Methodology

### Data Preprocessing

- Removed duplicate rows; confirmed no missing values in the dataset
- Applied Label Encoding to categorical columns (`From Bank`, `To Bank`, `From Account`, `To Account`, `Receiving Currency`, `Payment Currency`)
- Applied ordinal encoding to `Payment Format` based on transaction complexity hierarchy
- Converted `Timestamp` to Unix epoch (seconds) for numerical comparability
- Applied `MinMaxScaler` to all numerical features to normalise to [0, 1]

### Class Imbalance Handling

With a ~1,000:1 class ratio, standard training heavily biases predictions toward the majority class. Random undersampling was applied to create a balanced dataset:

- Suspicious transactions (class 1): 4,281 samples
- Legitimate transactions (class 0): 4,281 samples (randomly sampled from majority class)
- Total balanced dataset: **8,562 samples**

Stratified K-Fold (5 splits) was used for cross-validation to preserve class proportions across folds.

### Exploratory Analysis

Correlation heatmaps were computed for both the imbalanced and balanced datasets to surface feature-label relationships:

- `Timestamp` shows a positive correlation with laundering labels — later timestamps are more likely to be associated with suspicious activity
- `Payment Format` shows a negative correlation — simpler payment methods (lower ordinal values) are more associated with laundering

Outlier removal was applied to the `Timestamp` feature using the IQR method (1.5× IQR cutoff), reducing the influence of extreme values on model training.

### Dimensionality Reduction (Visualisation)

Three dimensionality reduction techniques were applied to the balanced dataset for 2D visualisation and class separability analysis:

| Method | Observation |
|---|---|
| PCA | Classes are not linearly separable — confirms need for non-linear models |
| t-SNE | Some local clustering visible but classes heavily overlap |
| TruncatedSVD | Similar to PCA; confirms non-linear structure |
| t-SNE + PCA (combined) | Slight improvement in cluster separation but still overlapping |

### Models Benchmarked

Five classifiers were trained and evaluated on the balanced undersampled dataset with 80/20 train-test split:

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline |
| Support Vector Classifier (SVC) | Kernel-based |
| K-Nearest Neighbours | Distance-based |
| Decision Tree | Single tree baseline |
| Random Forest | Ensemble baseline |
| XGBoost | Final model — GridSearchCV hyperparameter tuning |

### XGBoost Hyperparameter Tuning

GridSearchCV was used to tune the final XGBoost model with the following parameter grid optimising for F1 score on the balanced test set.

---

## Results

**Final Model: XGBoost (tuned)**

| Metric | Value |
|---|---|
| Accuracy | 89% |
| F1-Score | 0.85 |

**Key findings from EDA:**

- `Timestamp` is positively correlated with laundering labels — transactions occurring later in the observed period are more likely to be flagged as suspicious
- `Payment Format` is negatively correlated — simpler, lower-complexity payment methods are more associated with laundering activity
- PCA visualisation confirmed the data is not linearly separable, validating the choice of ensemble tree methods over linear classifiers

---

## Limitations & Future Work

**Current Limitations:**

- Undersampling discards the vast majority of legitimate transaction data — while necessary to address the class imbalance, it reduces the diversity of negative examples available during training
- The model is trained on a simulated IBM dataset; performance on real-world financial transaction data may differ due to distribution shift
- No temporal validation was performed — a time-based train/test split (training on earlier transactions, testing on later ones) would better simulate real deployment conditions
- Feature engineering on graph-based relationships (e.g., account network centrality, transaction chain patterns) was not explored, which could significantly improve detection of layering and integration patterns

**Future Directions:**

- Implement graph-based features using transaction networks modelled in Neo4j or NetworkX, capturing multi-hop laundering patterns that tabular features cannot represent
- Explore SMOTE or ADASYN as alternatives to undersampling, preserving more of the majority class signal
- Apply time-series cross-validation to evaluate model robustness across different time windows
- Investigate explainability via SHAP values to surface which features drive individual suspicious transaction predictions — critical for regulatory compliance and auditing

---

## How to Run This Project

### Prerequisites

```bash
Python 3.8+
```

### 1. Clone the Repository

```bash
git clone https://github.com/d4h2nu8h/aml-transaction-detection.git
cd aml-transaction-detection
```

### 2. Download the Dataset

Download `HI-Small_Trans.csv` from the [IBM AML Dataset on Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) and place it in the project directory.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Key libraries:

```bash
pip install xgboost category_encoders scikit-learn pandas numpy matplotlib seaborn plotly
```

### 4. Run the Notebook

```bash
jupyter notebook Money_Laundering_Patterns_Using_Financial_Transactions_Data.ipynb
```

Run all cells sequentially. The notebook covers preprocessing, EDA, dimensionality reduction, model benchmarking, and final XGBoost evaluation.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Dimensionality Reduction | PCA, t-SNE, TruncatedSVD (scikit-learn) |
| Data Processing | Pandas, NumPy, category_encoders |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Notebook Environment | Jupyter Notebook / Google Colab |
| Dataset Source | Kaggle — IBM AML Dataset |

---

## Author

**Dhanush Sambasivam**

[![GitHub](https://img.shields.io/badge/GitHub-d4h2nu8h-181717?style=flat&logo=github)](https://github.com/d4h2nu8h)

---

## License

This project is intended for academic and research purposes. Dataset sourced from the IBM Transactions for Anti-Money Laundering dataset, publicly available on Kaggle.
