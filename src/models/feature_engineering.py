import os
import pandas as pd
from optbinning import BinningProcess

PROCESSED_DATA_DIR = "data/processed"

def run_feature_engineering():
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "test.parquet"))
    
    # Exclude non-predictor columns and dynamic tracking fields from WoE fitting
    exclude_cols = ["target", "issue_d", "loan_amnt", "int_rate"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    
    print(f"Fitting Optimal Binning across {len(feature_cols)} features...")
    binning_process = BinningProcess(
        variable_names=feature_cols,
        min_iv=0.02,        # Prune noise features where IV < 0.02
        max_n_bins=8        # Standard coarse bin boundary limit
    )
    
    binning_process.fit(X_train, y_train)
    
    # Inspect Information Value (IV) summary
    iv_summary = binning_process.summary()
    print("\n--- Information Value (IV) Selection Matrix ---")
    print(iv_summary[["name", "iv", "quality_score"]])
    
    # Transform to WoE log-odds linear space
    X_train_woe = binning_process.transform(X_train)
    X_test_woe = binning_process.transform(test_df[feature_cols])
    
    # Re-attach target and pricing metadata for downstream evaluation
    X_train_woe["target"] = y_train.values
    X_train_woe["loan_amnt"] = train_df["loan_amnt"].values
    X_train_woe["int_rate"] = train_df["int_rate"].values
    
    X_test_woe["target"] = test_df["target"].values
    X_test_woe["loan_amnt"] = test_df["loan_amnt"].values
    X_test_woe["int_rate"] = test_df["int_rate"].values
    X_test_woe["issue_d"] = test_df["issue_d"].values
    
    X_train_woe.to_parquet(os.path.join(PROCESSED_DATA_DIR, "train_woe.parquet"), index=False)
    X_test_woe.to_parquet(os.path.join(PROCESSED_DATA_DIR, "test_woe.parquet"), index=False)
    print("\nWoE transformed datasets saved successfully.")

if __name__ == "__main__":
    run_feature_engineering()
