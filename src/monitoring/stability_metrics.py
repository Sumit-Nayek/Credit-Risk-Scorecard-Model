import numpy as np
import pandas as pd

def calculate_psi(reference: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
    """
    Computes the Population Stability Index (PSI) between a baseline dataset and actual production data.
    
    PSI Rule of Thumb:
    PSI < 0.1: No significant change; stable population distribution.
    0.1 <= PSI < 0.25: Moderate shift; alert risk team to monitor carefully.
    PSI >= 0.25: Major population drift; require immediate model recalibration.
    """
    # Use quantiles from the reference dataset to establish stable boundary lines
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(reference, quantiles)
    
    # Adjust boundaries slightly to catch edge values safely
    bins[0] -= 1e-5
    bins[-1] += 1e-5
    
    # Calculate frequency counts across both populations
    ref_counts, _ = np.histogram(reference, bins=bins)
    target_counts, _ = np.histogram(target, bins=bins)
    
    # Convert counts to actual percentages
    ref_pcts = ref_counts / len(reference)
    target_pcts = target_counts / len(target)
    
    # Apply a minor adjustment to avoid divide-by-zero errors on empty bins
    ref_pcts = np.where(ref_pcts == 0, 1e-4, ref_pcts)
    target_pcts = np.where(target_pcts == 0, 1e-4, target_pcts)
    
    # Core mathematical formula for PSI tracking
    psi_value = np.sum((target_pcts - ref_pcts) * np.log(target_pcts / ref_pcts))
    return float(psi_value)

if __name__ == "__main__":
    print("[-] Simulating production monitoring check...")
    # Generate mock data representing a stable environment vs a macro-drift scenario
    np.random.seed(42)
    baseline_scores = np.random.normal(650, 50, 5000)
    stable_production_scores = np.random.normal(648, 51, 5000)
    drifted_production_scores = np.random.normal(590, 65, 5000) # Significant economic stress scenario
    
    stable_psi = calculate_psi(baseline_scores, stable_production_scores)
    drifted_psi = calculate_psi(baseline_scores, drifted_production_scores)
    
    print("\n================== PRODUCTION MONITORING DASHBOARD ==================")
    print(f"Stable Population Metric (Week 1 Monitoring): PSI = {stable_psi:.4f}")
    if stable_psi < 0.1:
        print(" -> System Status: GREEN. Population profile matches baseline guidelines.")
        
    print(f"\nDrifted Population Metric (Macro Economic Shock): PSI = {drifted_psi:.4f}")
    if drifted_psi >= 0.25:
        print(" -> System Status: RED CRITICAL ALERT. Require immediate model recalibration.")
    print("=====================================================================\n")