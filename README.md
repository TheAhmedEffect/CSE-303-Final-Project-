# Loan Default Prediction

Classification of loan default on a 485-row, 8-feature credit dataset, built for
CSE-303 Statistics for Data Science. Python, pandas, scikit-learn.

Run `loan_default_prediction.ipynb`; the dataset sits beside it.

## What the notebook does

- **Load and inspect** — `read_csv`, `info`, `describe`, class balance, correlation matrix.
- **Explore** — distributions of income, debt-to-income and credit debt against the
  default flag (seaborn).
- **Split** — 80/20 train/test, stratified on the target, fixed `random_state`.
- **Scale** — `StandardScaler` fit on the training set only and applied to the test
  set, so no test-set statistics leak into training.
- **Model** — logistic regression and an RBF-kernel SVM, both with probability output.
- **Evaluate** — classification report, ROC-AUC, and ROC curves for both models.
- **Check for overfitting** — train AUC against test AUC for each model, plotted
  together rather than reporting test performance alone.

## Results

| Model | Test ROC-AUC | Train AUC | Accuracy | Recall (default class) |
|---|---|---|---|---|
| Logistic regression | **0.873** | 0.845 | 0.80 | 0.38 |
| SVM (RBF) | 0.811 | 0.874 | 0.77 | 0.21 |

## What the results actually say

Two things worth stating rather than hiding.

The simpler model won. Logistic regression scored a higher test AUC than the SVM and
showed no overfitting — its train AUC (0.845) was slightly *below* its test AUC
(0.873). The SVM went the other way, 0.874 on train against 0.811 on test, which is
the signature of a model fitting the training set harder than it generalises.

The headline accuracy is misleading. At 80% accuracy the logistic model still recalls
only 38% of actual defaults, because roughly three quarters of the dataset are
non-defaults and accuracy rewards predicting the majority class. On this problem
recall on the minority class is the number that matters, and neither model is good
enough at it. The obvious next steps are class weighting or resampling, and moving
the decision threshold off 0.5 to trade precision for recall deliberately.
