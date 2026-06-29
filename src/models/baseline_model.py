import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"

def run_baseline_training():
    print("[-] Loading partitioned datasets for baseline training...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    # Isolate targets from predictors
    X_train = train_df.drop(columns=['default_flag'])
    y_train = train_df['default_flag']
    X_test = test_df.drop(columns=['default_flag'])
    y_test = test_df['default_flag']
    
    print("[-] Fitting a standard, uncalibrated Logistic Regression model...")
    # Increase max_iter to guarantee convergence on unscaled raw metrics
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # Extract prediction probability curves
    train_preds = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model health via ROC-AUC (The industry index for separation power)
    train_auc = roc_auc_score(y_train, train_preds)
    test_auc = roc_auc_score(y_test, test_preds)
    
    print("\n================== BASELINE PERFORMANCE PERFORMANCES ==================")
    print(f"Train ROC-AUC Score: {train_auc:.4f}")
    print(f"Test ROC-AUC Score:  {test_auc:.4f}")
    print("=======================================================================\n")
    
    # Direct class outputs classification matrix
    hard_preds = model.predict(X_test)
    print(classification_report(y_test, hard_preds))

if __name__ == "__main__":
    run_baseline_training()