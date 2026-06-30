import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from optbinning import BinningProcess, Scorecard
from sklearn.metrics import roc_auc_score

# Structural paths
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"
SCORECARD_OUTPUT_PATH = "data/processed/scorecard_table.csv"

def run_scorecard_calibration():
    print("[-] Loading raw partitioned training profiles...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    X_train = train_df.drop(columns=['default_flag'])
    y_train = train_df['default_flag']
    X_test = test_df.drop(columns=['default_flag'])
    y_test = test_df['default_flag']
    
    # Define our 14 validated predictive features from Week 2
    selected_features = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3'
    ]
    
    print(f"[-] Initializing binning constraints for the {len(selected_features)} selected features...")
    binning_process = BinningProcess(variable_names=selected_features)
    
    # Establish production L2-penalized Logistic Regression
    log_reg = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    
    print("[-] Building enterprise scorecard using FICO min_max (300-850) configurations...")
    scorecard = Scorecard(
        binning_process=binning_process,
        estimator=log_reg,
        scaling_method="min_max",
        scaling_method_params={
            "min": 300,
            "max": 850
        }
    )
    
    # Fit the entire pipeline (Binning -> WoE -> Regression -> Point Calibration)
    scorecard.fit(X_train[selected_features], y_train)
    
    # Evaluate predictive power curves
    train_scores = scorecard.score(X_train[selected_features])
    test_scores = scorecard.score(X_test[selected_features])
    
    # Higher credit scores mean lower risk of default, so we invert the scores for AUC tracking
    train_auc = roc_auc_score(y_train, -train_scores)
    test_auc = roc_auc_score(y_test, -test_scores)
    
    print("\n================== PRODUCTION SCORECARD METRICS ==================")
    print(f"Calibrated Train ROC-AUC: {train_auc:.4f}")
    print(f"Calibrated Test ROC-AUC:  {test_auc:.4f}")
    print("===================================================================\n")
    
    # FIX: Explicitly pass style="detailed" to include WoE and Coefficient columns
    scorecard_df = scorecard.table(style="detailed")
    
    print("[-] Displaying slice of the calibrated point assignment matrix...")
    print(scorecard_df[['Variable', 'Bin', 'WoE', 'Coefficient', 'Points']].head(15).to_string(index=False))
    
    # Save the full detailed sheet to disk for our Week 4 API to look up
    os.makedirs(os.path.dirname(SCORECARD_OUTPUT_PATH), exist_ok=True)
    scorecard_df.to_csv(SCORECARD_OUTPUT_PATH, index=False)
    print(f"\n[+] Production scorecard lookup matrix archived at: {SCORECARD_OUTPUT_PATH}")

if __name__ == "__main__":
    run_scorecard_calibration()