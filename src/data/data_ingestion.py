import os
import getpass
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_SLUG = "adarshsng/lending-club-loan-data-csv"
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

def setup_kaggle_auth():
    """Prompt for credentials at runtime if not already set."""
    if not os.environ.get("KAGGLE_USERNAME") or not os.environ.get("KAGGLE_KEY"):
        print("--- Kaggle API Authentication ---")
        os.environ["KAGGLE_USERNAME"] = input("Enter Kaggle Username: ").strip()
        os.environ["KAGGLE_KEY"] = getpass.getpass("Enter Kaggle API Key: ").strip()

def download_and_extract():
    """Download the LendingClub dataset and unzip it."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    setup_kaggle_auth()
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()
    
    print(f"Downloading {DATASET_SLUG}...")
    api.dataset_download_files(DATASET_SLUG, path=RAW_DATA_DIR, unzip=True)
    print("Download and extraction complete.")

def process_and_partition():
    """Ingest raw CSV, filter completed loans, and perform stratified 70/30 split."""
    csv_path = os.path.join(RAW_DATA_DIR, "loan.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected file at {csv_path}")

    print("Loading and filtering loan records...")
    # Select candidate features at origination and outcomes
    selected_cols = [
        "loan_status", "loan_amnt", "term", "int_rate", "installment",
        "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "purpose", "dti", "delinq_2yrs", "inq_last_6mths",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "issue_d"
    ]
    
    df = pd.read_csv(csv_path, usecols=selected_cols, low_memory=False)
    
    # Filter completed loan cycles: 0 = Fully Paid, 1 = Default / Charged Off
    df = df[df["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
    df["target"] = (df["loan_status"] == "Charged Off").astype(int)
    df.drop(columns=["loan_status"], inplace=True)
    
    print(f"Dataset Shape: {df.shape} | Default Rate: {df['target'].mean():.2%}")

    # Stratified 70/30 Train/Test split
    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["target"]
    )

    train_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "test.parquet"), index=False)
    print("Train/Test sets saved successfully to data/processed/ as Parquet.")

if __name__ == "__main__":
    download_and_extract()
    process_and_partition()
