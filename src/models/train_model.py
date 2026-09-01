import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

PROCESSED_DATA_DIR = "data/processed"

def train_and_calibrate():
    print("Loading WoE transformed data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train_woe.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "test_woe.parquet"))

    # Metadata columns to exclude from training
    non_feature_cols = ["target", "loan_amnt", "int_rate", "issue_d"]
    feature_cols = [c for c in train_df.columns if c not in non_feature_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    print(f"Fitting Regularized Logistic Regression on {len(feature_cols)} features...")
    model = LogisticRegression(penalty="l2", C=0.1, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Performance evaluation
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    print(f"\nModel Performance:")
    print(f"Train ROC-AUC: {train_auc:.4f}")
    print(f"Test ROC-AUC:  {test_auc:.4f}")

    # Scorecard Calibration Parameters
    # Target: 600 Score at 50:1 Good-to-Bad Odds (P_default = 1 / 51 ≈ 0.0196)
    # PDO (Points to Double Odds) = 20
    s_0 = 600.0
    pdo = 20.0
    odds_0 = 50.0  # Good:Bad = 50:1 => ln(Odds) = ln(50)

    factor = pdo / np.log(2)
    offset = s_0 - (factor * np.log(odds_0))
    print(f"\nCalibration Constants: Factor = {factor:.4f}, Offset = {offset:.4f}")

    # Compute applicant log-odds: ln(Good/Bad) = - (beta_0 + sum(beta_j * WoE_j))
    # Standard credit scoring convention uses Good-to-Bad log-odds for points scaling
    log_odds_train = -(model.intercept_[0] + np.dot(X_train.values, model.coef_[0]))
    log_odds_test = -(model.intercept_[0] + np.dot(X_test.values, model.coef_[0]))

    # Scale directly to 300 - 850 point range
    train_scores = np.clip(np.round(offset + factor * log_odds_train), 300, 850).astype(int)
    test_scores = np.clip(np.round(offset + factor * log_odds_test), 300, 850).astype(int)

    # Attach credit scores and probabilities to test set for downstream business analytics
    test_df["credit_score"] = test_scores
    test_df["default_proba"] = test_preds

    test_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "test_scored.parquet"), index=False)
    print(f"Scored test records saved to data/processed/test_scored.parquet")
    print(f"Score Range: Min = {test_scores.min()}, Median = {np.median(test_scores):.0f}, Max = {test_scores.max()}")

if __name__ == "__main__":
    train_and_calibrate()