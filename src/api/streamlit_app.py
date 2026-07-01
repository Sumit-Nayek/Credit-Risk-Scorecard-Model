import os
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Credit Risk Decision Engine", layout="centered")

st.title("🛡️ Enterprise Credit Risk Scorecard Engine")
st.markdown("""
This production-grade interface evaluates consumer application risk using an institutional 
**Weight of Evidence (WoE) Scorecard**. Adjust the financial profile parameters below to simulate real-time underwriting decisions.
""")

SCORECARD_PATH = "data/processed/scorecard_table.csv"

@st.cache_data
def load_scorecard():
    if os.path.exists(SCORECARD_PATH):
        return pd.read_csv(SCORECARD_PATH)
    return None

scorecard_df = load_scorecard()

def lookup_points(variable_name: str, value: float) -> float:
    if scorecard_df is None:
        return 46.4
    var_rules = scorecard_df[scorecard_df['Variable'] == variable_name]
    for _, row in var_rules.iterrows():
        bin_str = str(row['Bin'])
        if bin_str in ["Special", "Missing"]:
            continue
        cleaned = bin_str.replace("(", "").replace("[", "").replace(")", "").replace("]", "")
        parts = cleaned.split(",")
        lower = float(parts[0].strip())
        upper = float(parts[1].strip())
        if lower <= value < upper:
            return float(row['Points'])
    return float(var_rules[var_rules['Bin'] == 'Missing']['Points'].fillna(46.4).iloc[0])

# --- SIDEBAR: APPLICANT PROFILE INPUTS ---
st.sidebar.header("Applicant Financial Profile")

limit_bal = st.sidebar.slider("Requested Credit Limit ($)", 10000, 500000, 100000, step=5000)
age = st.sidebar.slider("Applicant Age (Years)", 18, 80, 35, step=1)
sex = st.sidebar.selectbox("Gender Identification", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
education = st.sidebar.selectbox("Education Level", options=[1, 2, 3, 4], format_func=lambda x: ["Graduate School", "University", "High School", "Other"][x-1])
marriage = st.sidebar.selectbox("Marital Status", options=[1, 2, 3], format_func=lambda x: ["Married", "Single", "Other"][x-1])

st.sidebar.header("Payment & Bureau History")
pay_0 = st.sidebar.selectbox("Recent Repayment Status (Current Month)", options=[-1, 0, 1, 2, 3, 4], 
                             format_func=lambda x: "Paid Up / No Debt" if x <= 0 else f"{x} Month(s) Delinquent")
pay_amt1 = st.sidebar.slider("Previous Amount Paid (Month 1) ($)", 0, 20000, 3000, step=500)
pay_amt2 = st.sidebar.slider("Previous Amount Paid (Month 2) ($)", 0, 20000, 3000, step=500)

# --- UNDERWRITING CORE ENGINE EXECUTION ---
applicant_payload = {
    'LIMIT_BAL': limit_bal, 'SEX': sex, 'EDUCATION': education, 'MARRIAGE': marriage, 'AGE': age,
    'PAY_0': pay_0, 'PAY_2': 0, 'PAY_3': 0, 'PAY_4': 0, 'PAY_5': 0, 'PAY_6': 0,
    'PAY_AMT1': pay_amt1, 'PAY_AMT2': pay_amt2, 'PAY_AMT3': 2000
}

st.subheader("Automated Underwriting Analysis")

# 1. Hard Knock-Out Check
if pay_0 >= 3:
    st.error("❌ APPLICATION STATUS: REJECTED")
    st.metric(label="Calibrated Credit Score", value="N/A")
    st.warning("**Reason:** Hard Knock-Out Rule Triggered. Applicant exhibits critical immediate account delinquency (90+ days past due).")
else:
    # 2. Score Computation
    total_score = 0.0
    for variable, value in applicant_payload.items():
        total_score += lookup_points(variable, value)
    final_score = int(round(total_score))
    
    # 3. Cutoff Threshold Application (Optimized at 600 in Week 5)
    if final_score < 600:
        st.error(f"❌ APPLICATION STATUS: REJECTED (Score: {final_score})")
        st.metric(label="Calibrated Credit Score", value=final_score, delta="- Risk Variance")
        st.info("**Underwriting Note:** Score falls below the portfolio optimization profitability cutoff barrier (< 600).")
    else:
        st.success("✅ APPLICATION STATUS: APPROVED")
        
        # Split metrics grid
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Calibrated Credit Score", value=final_score)
        with col2:
            tier = "Super-Prime Tier" if final_score >= 650 else "Mid-Prime Tier"
            st.metric(label="Risk Classification", value=tier)
            
        # Pricing Assignment
        if final_score >= 650:
            assigned_limit = round(limit_bal * 1.2, -2)
            apr = "11.25%"
        else:
            assigned_limit = round(limit_bal * 0.5, -2)
            apr = "24.99%"
            
        st.markdown("### Institutional Pricing Offer Parameters")
        st.json({
            "Assigned Risk Credit Limit": f"${assigned_limit:,.2f}",
            "Offered Portfolio APR": apr,
            "Target Loss Given Default Bound (LGD)": "45.00%"
        })

# --- STAKEHOLDER FRAMEWORK FOOTER ---
st.markdown("---")
st.caption("🔒 MLOps Enterprise Architecture | Baseline Training Core Architecture Tracked via Git Control")