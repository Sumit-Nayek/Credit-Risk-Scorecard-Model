import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Enterprise configuration paths
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
RAW_DATA_PATH = "data/raw/credit_card_default.xlsx"
TRAIN_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"

def download_raw_data():
    """Downloads raw data if it doesn't already exist locally."""
    if not os.path.exists(RAW_DATA_PATH):
        print("[-] Downloading real-life credit default data from UCI repository...")
        df = pd.read_excel(DATA_URL, header=1)
        os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
        df.to_excel(RAW_DATA_PATH, index=False)
        print(f"[+] Raw data successfully archived at {RAW_DATA_PATH}")
    else:
        print("[*] Raw dataset already exists. Skipping download.")

def process_and_split_data():
    """Cleans basic structural flaws and executes a repeatable 70/30 stratified split."""
    print("[-] Reading raw data for extraction, transformation, and load (ETL)...")
    df = pd.read_excel(RAW_DATA_PATH)
    
    # Standardizing target variable name to fit credit nomenclature
    df = df.rename(columns={'default payment next month': 'default_flag'})
    
    # Drop the internal ID tracking column to avoid artificial predictive power
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    print(f"[*] Full Dataset Shape: {df.shape} | Base Default Rate: {df['default_flag'].mean():.2%}")
    
    # 70/30 Stratified Split ensures equal default distribution across both sets
    print("[-] Executing stratified data split...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.30, 
        stratify=df['default_flag'], 
        random_state=42
    )
    
    os.makedirs(os.path.dirname(TRAIN_DATA_PATH), exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"[+] Processed train set ({train_df.shape}) saved to {TRAIN_DATA_PATH}")
    print(f"[+] Processed test set ({test_df.shape}) saved to {TEST_DATA_PATH}")

if __name__ == "__main__":
    download_raw_data()
    process_and_split_data()