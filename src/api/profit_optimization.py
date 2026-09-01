import os
import pandas as pd
import numpy as np

PROCESSED_DATA_DIR = "data/processed"

def run_profit_simulation():
    scored_path = os.path.join(PROCESSED_DATA_DIR, "test_scored.parquet")
    if not os.path.exists(scored_path):
        raise FileNotFoundError(f"Missing scored file at {scored_path}. Run train_model.py first.")

    print("Loading scored test portfolio for financial simulation...")
    df = pd.read_parquet(scored_path)

    total_test_records = len(df)
    lgd_penalty = 0.45  # Loss Given Default = 45% of principal lost on default

    cutoff_scores = range(520, 680, 20)
    results = []

    for cutoff in cutoff_scores:
        # Strategy rule: Approve applicants with score >= cutoff
        approved = df[df["credit_score"] >= cutoff]
        approved_count = len(approved)

        if approved_count == 0:
            continue

        # Non-Performing Loan (NPL) Default Rate
        defaults = approved[approved["target"] == 1]
        goods = approved[approved["target"] == 0]
        npl_rate = len(defaults) / approved_count

        # Financial Equation:
        # Profit = Interest Revenue from Goods - Principal Losses from Bads
        interest_revenue = (goods["loan_amnt"] * (goods["int_rate"] / 100.0)).sum()
        principal_loss = (defaults["loan_amnt"] * lgd_penalty).sum()
        net_profit = interest_revenue - principal_loss

        results.append({
            "Cutoff Score": cutoff,
            "Approval Rate": f"{(approved_count / total_test_records):.2%}",
            "Approved Count": f"{approved_count:,} / {total_test_records:,}",
            "NPL Default Rate": f"{npl_rate:.2%}",
            "Net Profit (USD)": f"${net_profit:,.2f}",
            "_raw_profit": net_profit
        })

    sim_table = pd.DataFrame(results)
    
    # Identify optimal boundary
    optimal_idx = sim_table["_raw_profit"].idxmax()
    optimal_row = sim_table.loc[optimal_idx]

    print("\n--- Portfolio Profit Optimization Simulation Matrix ---")
    print(sim_table.drop(columns=["_raw_profit"]).to_string(index=False))

    print(f"\nOptimal Cutoff Boundary: {optimal_row['Cutoff Score']} Points")
    print(f"Max Net Test Revenue:   {optimal_row['Net Profit (USD)']}")
    print(f"Controlled NPL Rate:    {optimal_row['NPL Default Rate']}")

if __name__ == "__main__":
    run_profit_simulation()