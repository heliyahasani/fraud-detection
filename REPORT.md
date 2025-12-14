# Fraud Detection Model - Final Report

## Dataset Overview

| Metric             | Value         |
| ------------------ | ------------- |
| Total transactions | 1,852,394     |
| Fraud cases        | 9,651 (0.52%) |
| Test set           | 370,479 rows  |
| Test frauds        | 1,930         |

---

## The Problem

Without handling class imbalance, baseline logistic regression achieves 99% accuracy but catches only **3.1% of fraud** (59 out of 1,930). The model learns to predict everything as legit because that's the easy path to high accuracy.

---

## Training Decisions

### Class Imbalance: Class Weights vs SMOTE

Used PR-AUC to compare (not accuracy or F1) because with 0.52% fraud:

- Accuracy is useless - predicting all legit gives 99.5%
- F1 depends on threshold choice - unfair comparison
- PR-AUC measures ranking ability across all thresholds, focuses on minority class

| Approach      | PR-AUC | Decision     |
| ------------- | ------ | ------------ |
| Class Weights | 0.207  | **Selected** |
| SMOTE         | 0.185  | Rejected     |

Went with class weights - simpler, no synthetic data overhead, and slightly better PR-AUC.

### Model Selection

Compared 4 models on 10k stratified sample (2-fold CV):

| Model         | PR-AUC |
| ------------- | ------ |
| XGBoost       | 0.463  |
| Random Forest | 0.460  |
| Logistic      | 0.247  |
| Neural Net    | 0.189  |

XGBoost wins. Not surprising I think the reason is gradient boosting handles imbalanced tabular data well.

### Final Model Parameters

After tuning on full training data:

```
CV PR-AUC: 0.8664

n_estimators: 200
max_depth: 7
learning_rate: 0.1
scale_pos_weight: 10
reg_alpha: 0
reg_lambda: 1
```

Minimal regularization needed. The model generalizes fine without heavy constraints.

### Overfitting & Regularization

Checked for overfitting using learning curves and validation curves on all 4 models.

**What we looked for:**

- Learning curves: if train score >> validation score = overfitting
- Validation curves: how hyperparameters affect the train/val gap

**What we found:**

- XGBoost and Random Forest showed small gaps (~0.1) between train and validation - acceptable
- Logistic regression underfits (both scores low) - too simple for this problem
- Neural net had higher variance but no severe overfitting

**Validation curve optimal values (10k sample, single param tested):**

| Model         | Param     | Optimal | PR-AUC |
| ------------- | --------- | ------- | ------ |
| Logistic      | C         | 0.01    | 0.253  |
| Random Forest | max_depth | 15      | 0.456  |
| XGBoost       | max_depth | 3       | 0.485  |
| Neural Net    | alpha     | 0.0001  | 0.189  |

Note: Final tuned XGBoost has max_depth=7, not 3.Grid search tests combinations - max_depth=7 works better with other params (learning_rate, n_estimators, etc.) on full data.

**Regularization in final model:**

- `reg_alpha: 0` (L1) - no sparsity needed, all features useful
- `reg_lambda: 1` (L2) - default, slight weight penalty
- `max_depth: 7` - limits tree complexity
- `scale_pos_weight: 10` - handles imbalance, not regularization but helps

**Data sizes at each step:**

| Step                       | Data Size | Why                                              |
| -------------------------- | --------- | ------------------------------------------------ |
| Model comparison           | 10k       | Fast iteration, find winning model type          |
| Learning/validation curves | 10k       | Diagnostic plots, check overfitting patterns     |
| Hyperparameter tuning      | 1.48M     | Full training data for best params (XGBoost won) |
| Final evaluation           | 370k      | Held-out test set, never seen during training    |

We didn't need heavy regularization because:

1. Tuned on 1.48M samples - plenty of data for 16 features
2. XGBoost's built-in early stopping prevents overfitting
3. Features are already engineered (no raw high-cardinality stuff)

CV PR-AUC (0.87) close to test PR-AUC (0.89) confirms no overfitting.

---

## Evaluation Results

Evaluated on held-out test set (370,479 transactions, 1,930 frauds).

### Performance Summary

| Metric    | Value  | What it means                                                     |
| --------- | ------ | ----------------------------------------------------------------- |
| PR-AUC    | 0.8939 | Model's ranking ability (1.0 = perfect, random = 0.005)           |
| ROC-AUC   | 0.9971 | Overall discrimination (high due to class imbalance, less useful) |
| Precision | 71.7%  | Of transactions flagged as fraud, 71.7% are actually fraud        |
| Recall    | 86.2%  | Of actual frauds, we catch 86.2%                                  |
| F1        | 0.7830 | Harmonic mean of precision and recall                             |
| Threshold | 0.50   | If model probability > 0.5, flag as fraud                         |

### Confusion Matrix

```
                 Predicted
              Legit    Fraud
Actual Legit  367,894    655    ← False alarms (0.18% of legit)
       Fraud     267   1,663    ← Caught 86.2% of fraud
```

**What this means:**

- **367,894 TN**: Legit transactions correctly left alone
- **655 FP**: Legit transactions wrongly flagged (false alarms - customers get annoyed)
- **267 FN**: Fraud we missed (bad - money lost)
- **1,663 TP**: Fraud we caught (good)

**Bottom line**: Catches 1,663 out of 1,930 frauds. Misses 267. Generates 655 false alarms out of 368k legit transactions.

### Top Features

| Feature           | Importance |
| ----------------- | ---------- |
| is_night          | 44.4%      |
| amt               | 19.6%      |
| is_online         | 8.7%       |
| cat_risk_medium   | 6.4%       |
| merch_risk_medium | 5.9%       |

Night transactions and amount dominate. Makes sense fraudsters often operate at night when victims are asleep.

---

## Error Analysis

### Missed Fraud (267 cases)

| Stat                 | Value    |
| -------------------- | -------- |
| Mean probability     | 0.20     |
| Max probability      | 0.49     |
| Close misses (>0.40) | 41 (15%) |

Most missed frauds have low probability scores the model is genuinely uncertain about them. 41 cases were close calls (proba 0.40-0.49), could catch these by lowering threshold but would increase false alarms.

### False Alarms (655 cases)

| Stat             | Value |
| ---------------- | ----- |
| Mean probability | 0.67  |
| Min probability  | 0.50  |

False alarms cluster just above the 0.5 threshold. These are borderline cases where legit transactions look suspicious.

---

## What's Not Great

1. **267 missed frauds** - Mean probability 0.20, max 0.49. Model is genuinely uncertain - these don't look like typical fraud. Hard cases.

2. **655 false alarms** - Mean probability 0.67, cluster just above threshold. Legit transactions that happen to match fraud patterns (night, high amount, etc).

3. **is_night dominates (44%)** - Simple binary feature carries too much weight. Could mean the model is overfitting to time-of-day patterns.

4. **No velocity features** - Can't detect "5 transactions in 10 minutes" pattern. This is a major fraud signal we're missing.

5. **Neural net failed (0.189 PR-AUC)** - Performed worse than random forest. Probably needs more data or different architecture for tabular data.

6. **ROC-AUC misleading (0.9971)** - Looks amazing but inflated by class imbalance. PR-AUC (0.89) is the honest metric.

---

## Potential Improvements

### Feature Engineering (highest impact)

- **Velocity features** - transactions per hour/day per card. Would catch burst patterns.
- **Distance velocity** - how fast is the card "traveling" between transactions
- **Better time features** - day of week, weekend flag, hour buckets instead of just is_night
- **Transaction sequence** - is this card's spending pattern unusual today?

### Model Improvements

- **Lower threshold for review queue** - Flag proba 0.35-0.50 for human review. Would catch 41 more frauds.
- **Cost-sensitive learning** - Missing fraud costs more than false alarm. Adjust accordingly.
- **Ensemble with Isolation Forest** - Catch outlier fraud patterns that don't fit supervised model.

### Production Considerations

- **Human review queue** - Borderline predictions (0.35-0.50) go to analysts instead of auto-block
- **Monitor for drift** - Fraud patterns change. Retrain monthly or when metrics drop.
- **A/B test thresholds** - Different thresholds for different transaction amounts

---

## Files

| File                            | Description             |
| ------------------------------- | ----------------------- |
| `models/fraud_detector.joblib`  | Trained XGBoost model   |
| `models/model_config.txt`       | Model parameters        |
| `notebooks/03_Training.ipynb`   | Training pipeline       |
| `notebooks/04_Evaluation.ipynb` | Evaluation and analysis |

Run `mlflow ui` in notebooks directory to view experiment tracking.
