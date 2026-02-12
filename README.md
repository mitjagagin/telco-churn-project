# Telco Customer Churn (classification)

## Goal
Predict customer churn (binary classification).

## Data
Kaggle Telco Customer Churn dataset (~7k rows, churn rate ~26.5%).

## Approach
- EDA + data cleaning (TotalCharges converted to numeric; customerID removed)
- Preprocessing with scikit-learn Pipeline + ColumnTransformer:
  - numeric: StandardScaler
  - categorical: OneHotEncoder(handle_unknown="ignore")
- Baseline model: LogisticRegression
- Threshold tuning by F1 to improve churn-class recall/F1

## Results
- ROC-AUC: 0.836
- F1 (thr=0.50): 0.608
- Best threshold: 0.40
- F1 (thr=0.40): 0.627
- Recall churn (thr=0.40): 0.68

## Links
- Kaggle notebook: https://www.kaggle.com/code/mitjagagin/telco-customer-churn-classification
