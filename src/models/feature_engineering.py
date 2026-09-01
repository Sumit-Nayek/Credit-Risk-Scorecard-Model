import os
import pandas as pd
from optbinning import BinningProcess

PROCESSED_DATA_DIR = "data/processed"

def run_feature_engineering():
    print("Loading training data...")
    train_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"))
    test_df = pd.read_parquet(os.path.join(PROCESSED_DATA_DIR, "test.parquet"))
    
    # Pre-clean known percentage columns if they were parsed as strings
    for df in [train_df, test_df]:
        for col in ["int_rate", "revol_util"]:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '').astype(float)
                
    # Define categorical variables explicitly for optimal binning
    cat_cols = ["term", "grade", "sub_grade", "emp_length", 
                "home_ownership", "verification_status", "purpose"]
                
    # Ensure categorical columns are strictly strings to handle NaNs seamlessly
    for col in cat_cols:
        train_df[col] = train_df[col].astype(str).replace('nan', 'Missing')
        test_df[col] = test_df[col].astype(str).replace('nan', 'Missing')
    
    # Exclude non-predictor columns and pricing fields from WoE fitting
    exclude_cols = ["target", "issue_d", "loan_amnt", "int_rate"]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    # Identify which of our feature columns are categorical
    actual_cat_cols = [col for col in feature_cols if col in cat_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    
    print(f"Fitting Optimal Binning across {len(feature_cols)} features...")
    binning_process = BinningProcess(
        variable_names=feature_cols,
        categorical_variables=actual_cat_cols,
        selection_criteria={"iv": {"min": 0.02}}, # Corrected IV threshold syntax
        max_n_bins=8,       
        n_jobs=-1           
    )
    
    binning_process.fit(X_train, y_train)
    
    # Inspect Information Value (IV) summary matrix
    iv_summary = binning_process.summary()
    print("\n--- Information Value (IV) Selection Matrix (Top Features) ---")
    print(iv_summary[["name", "iv", "status"]].head(10))
    
    # Save the selected feature names
    selected_features = binning_process.get_support(names=True)
    print(f"\nRetained {len(selected_features)} highly predictive features.")
    
    print("Transforming features to WoE log-odds space...")
    X_train_woe = binning_process.transform(X_train)
    X_test_woe = binning_process.transform(test_df[feature_cols])
    
    # Re-attach target and metadata necessary for downstream modules
    for df, woe_df in zip([train_df, test_df], [X_train_woe, X_test_woe]):
        woe_df["target"] = df["target"].values
        woe_df["loan_amnt"] = df["loan_amnt"].values
        woe_df["int_rate"] = df["int_rate"].values
        woe_df["issue_d"] = df["issue_d"].values
    
    X_train_woe.to_parquet(os.path.join(PROCESSED_DATA_DIR, "train_woe.parquet"), index=False)
    X_test_woe.to_parquet(os.path.join(PROCESSED_DATA_DIR, "test_woe.parquet"), index=False)
    print("\nWoE transformed datasets saved successfully to data/processed/")

if __name__ == "__main__":
    run_feature_engineering()