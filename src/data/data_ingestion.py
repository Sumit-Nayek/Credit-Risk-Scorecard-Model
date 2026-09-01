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
    """Ingest raw CSV in memory-efficient chunks, filter completed loans, and partition."""
    csv_path = os.path.join(RAW_DATA_DIR, "loan.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Expected file at {csv_path}")

    print("Loading and filtering loan records in chunks to manage memory...")
    selected_cols = [
        "loan_status", "loan_amnt", "term", "int_rate", "installment",
        "grade", "sub_grade", "emp_length", "home_ownership", "annual_inc",
        "verification_status", "purpose", "dti", "delinq_2yrs", "inq_last_6mths",
        "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc", "issue_d"
    ]
    
    chunks = []
    # Stream in chunks of 100,000 rows to keep RAM usage low
    for i, chunk in enumerate(pd.read_csv(csv_path, usecols=selected_cols, chunksize=100000, low_memory=False)):
        filtered_chunk = chunk[chunk["loan_status"].isin(["Fully Paid", "Charged Off"])].copy()
        filtered_chunk["target"] = (filtered_chunk["loan_status"] == "Charged Off").astype("int8")
        filtered_chunk.drop(columns=["loan_status"], inplace=True)
        chunks.append(filtered_chunk)
        print(f"Processed chunk {i+1}... Retained records: {sum(len(c) for c in chunks):,}")

    df = pd.concat(chunks, ignore_index=True)
    del chunks  # Free memory immediately
    
    print(f"\nFinal Dataset Shape: {df.shape} | Default Rate: {df['target'].mean():.2%}")

    print("Executing stratified 70/30 train/test partition...")
    train_df, test_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df["target"]
    )

    train_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "test.parquet"), index=False)
    print("Train/Test partitions successfully saved to data/processed/ as Parquet.")

if __name__ == "__main__":
    download_and_extract()
    process_and_partition()
