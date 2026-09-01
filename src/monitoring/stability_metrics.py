import os
import pandas as pd
import numpy as np

PROCESSED_DATA_DIR = "data/processed"

def calculate_psi(expected_scores, actual_scores, num_buckets=10):
    # Sort and define decile boundaries natively on the baseline population
    quantiles = np.linspace(0, 1, num_buckets + 1)
    bins = np.percentile(expected_scores, quantiles * 100)
    bins[0], bins[-1] = -np.inf, np.inf
    
    # Calculate densities across bins
    expected_counts, _ = np.histogram(expected_scores, bins=bins)
    actual_counts, _ = np.histogram(actual_scores, bins=bins)
    
    # Convert counts to percentages
    expected_pct = expected_counts / len(expected_scores)
    actual_pct = actual_counts / len(actual_scores)
    
    # Secure zero divisions
    eps = 1e-4
    expected_pct = np.where(expected_pct == 0, eps, expected_pct)
    actual_pct = np.where(actual_pct == 0, eps, actual_pct)
    
    # PSI vector calculus
    psi_vector = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi_total = np.sum(psi_vector)
    
    # Package output table
    psi_table = pd.DataFrame({
        "Score Range": [f"{bins[i]:.0f} to {bins[i+1]:.0f}" for i in range(len(bins)-1)],
        "Baseline %": expected_pct,
        "Current %": actual_pct,
        "PSI Bucket": psi_vector
    })
    
    return psi_total, psi_table

def run_stability_monitoring():
    scored_path = os.path.join(PROCESSED_DATA_DIR, "test_scored.parquet")
    if not os.path.exists(scored_path):
        raise FileNotFoundError(f"Missing scored file at {scored_path}.")

    df = pd.read_parquet(scored_path)
    
    print("Formatting chronological telemetry arrays...")
    # Clean datetime format for temporal partitioning
    df["issue_date"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_date", "credit_score"])
    
    # Baseline vs Current macroeconomic split (e.g., Pre-2016 vs 2016 Onward)
    baseline_df = df[df["issue_date"].dt.year <= 2015]
    current_df  = df[df["issue_date"].dt.year > 2015]
    
    print(f"Baseline Population (<=2015): {len(baseline_df):,} records")
    print(f"Current Population (>2015):   {len(current_df):,} records")
    
    psi_total, psi_table = calculate_psi(baseline_df["credit_score"], current_df["credit_score"])
    
    print("\n--- Population Stability Index (PSI) Drift Telemetry ---")
    print(psi_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSystem Total PSI: {psi_total:.4f}")
    
    # Base Basel MRM alert thresholds
    if psi_total < 0.10:
        print("\nStatus: GREEN | Model stable. No drift.")
    elif psi_total < 0.25:
        print("\nStatus: YELLOW | Moderate macroeconomic drift detected. Monitor closely.")
    else:
        print("\nStatus: RED | Critical Alert: Macroeconomic drift exceeding regulatory thresholds (>0.25). Recalibration required.")

if __name__ == "__main__":
    run_stability_monitoring()