#  Predicting Term Deposit Subscription Using Machine Learning

This project focuses on building machine learning models to predict whether a client will subscribe to a term deposit, using demographic, social, and financial data from the "Bank Marketing" dataset. The dataset, sourced from the UCI Machine Learning Repository, contains information from a Portuguese banking institution's direct marketing campaigns.

## Dataset

- **Source:** [UCI Machine Learning Repository - Bank Marketing Data Set](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Instances:** 45,211
- **Features:** 17 (mix of categorical and numerical)
- **Target:** `y` (whether the client subscribed to a term deposit)

## Project Overview

The project workflow includes:
- **Exploratory Data Analysis (EDA):** Understanding feature distributions, handling missing values (none found), and identifying categorical/numerical features.
- **Data Cleaning:** Outlier detection and handling using winsorization for features like `balance`, `duration`, `campaign`, `pdays`, and `previous`.
- **Feature Engineering:** 
  - Ordinal encoding for `education`
  - One-hot encoding for other categorical features
  - Standardization of numerical features for neural networks
- **Handling Class Imbalance:** SMOTE (Synthetic Minority Over-sampling Technique) is used to balance the training data.
- **Modeling:** Two main models were implemented:
  - **Random Forest (sklearn):** Hyperparameter tuning via GridSearchCV, class_weight set to handle imbalance.
  - **Neural Network (TensorFlow/Keras):** Sequential model with ReLU activations and a sigmoid output for binary classification.

## Results

| Metric         | Random Forest (Class 1) | Neural Network (Class 1) | Random Forest (Class 0) | Neural Network (Class 0) |
|----------------|------------------------|--------------------------|------------------------|--------------------------|
| Precision      | 0.63                   | 0.58                     | 0.94                   | 0.94                     |
| Recall         | 0.52                   | 0.56                     | 0.96                   | 0.95                     |
| F1-score       | 0.57                   | 0.57                     | 0.95                   | 0.94                     |
| Accuracy       | 0.91                   | 0.90                     |                        |                          |
| ROC-AUC        | 0.93                   | 0.92                     |                        |                          |

- Both models performed well, with Random Forest slightly outperforming Neural Networks on most metrics.
- The models are highly effective at identifying the majority class; performance on the minority class is lower due to class imbalance in the test set.

## Limitations

- **Model:** Random Forest is less interpretable and can overfit if not tuned. Neural Networks require careful tuning and are computationally intensive.
- **Data:** Significant class imbalance remains a challenge, even after applying SMOTE.

## Author

Chirath Setunge

