import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set publication-style visual parameters
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.autolayout': True,
    'figure.dpi': 300
})

os.makedirs("figures", exist_ok=True)

def plot_feature_importance():
    # Data from your IV generation terminal output
    features = ['sub_grade', 'grade', 'term', 'dti', 'verification_status', 
                'installment', 'home_ownership', 'annual_inc', 'purpose', 'emp_length']
    iv_scores = [0.4822, 0.4575, 0.1725, 0.0757, 0.0571, 0.0317, 0.0311, 0.0300, 0.0154, 0.0127]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=iv_scores, y=features, palette="viridis")
    plt.axvline(x=0.3, color='red', linestyle='--', label='Strong Predictive Threshold (0.3)')
    plt.axvline(x=0.02, color='gray', linestyle=':', label='Noise Threshold (0.02)')
    
    plt.title("Information Value (IV) by Feature (Explainability Matrix)")
    plt.xlabel("Information Value (IV)")
    plt.ylabel("Predictor Variable")
    plt.legend(loc="lower right")
    plt.savefig("figures/fig1_iv_matrix.pdf", bbox_inches='tight')
    print("Saved fig1_iv_matrix.pdf")

def plot_csuite_optimization():
    # Data from your Profit Optimization terminal output
    cutoffs = [520, 540, 560, 580]
    profits = [111508764.80, 79760970.11, 30675627.95, 4459369.98]
    profits_millions = [p / 1e6 for p in profits]
    npl_rates = [13.60, 8.78, 5.17, 2.84]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:green'
    ax1.set_xlabel('Scorecard Cutoff Threshold (Points)')
    ax1.set_ylabel('Net Portfolio Profit ($ Millions)', color=color)
    ax1.plot(cutoffs, profits_millions, marker='o', color=color, linewidth=2, label="Net Profit")
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Mark the sweet spot
    ax1.axvline(x=560, color='gray', linestyle='--', alpha=0.7)
    ax1.text(560.5, 50, 'Strategic Sweet Spot (560)', rotation=90, va='center', color='black')

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Non-Performing Loan (NPL) Rate %', color=color)  
    ax2.plot(cutoffs, npl_rates, marker='s', color=color, linewidth=2, linestyle=':', label="NPL Rate")
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Regulatory Warning Line
    ax2.axhline(y=7.0, color='darkred', linestyle='-', alpha=0.5, label="MRM Regulatory Limit (7%)")

    plt.title("Corporate Cost-Benefit Optimization Curve")
    fig.tight_layout()  
    plt.savefig("figures/fig2_profit_curve.pdf", bbox_inches='tight')
    print("Saved fig2_profit_curve.pdf")

def plot_psi_stability():
    # Data from your PSI Drift terminal output
    score_ranges = ['< 501', '501-511', '511-519', '519-525', '525-531', 
                    '531-538', '538-545', '545-553', '553-564', '> 564']
    baseline_pct = [9.65, 9.52, 10.62, 9.22, 9.57, 10.97, 10.23, 10.06, 10.00, 10.15]
    current_pct = [8.29, 8.95, 10.79, 9.81, 10.28, 11.44, 10.28, 9.58, 9.20, 11.38]

    x = np.arange(len(score_ranges))
    width = 0.35  

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, baseline_pct, width, label='Baseline (Pre-2016)', color='steelblue')
    ax.bar(x + width/2, current_pct, width, label='Current (Post-2015)', color='darkorange')

    ax.set_ylabel('Population Density (%)')
    ax.set_title('Macroeconomic Drift: Population Stability Index (Total PSI = 0.0058)')
    ax.set_xticks(x)
    ax.set_xticklabels(score_ranges, rotation=45)
    ax.legend()

    plt.savefig("figures/fig3_psi_drift.pdf", bbox_inches='tight')
    print("Saved fig3_psi_drift.pdf")

if __name__ == "__main__":
    plot_feature_importance()
    plot_csuite_optimization()
    plot_psi_stability()