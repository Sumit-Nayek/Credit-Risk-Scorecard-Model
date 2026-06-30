import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Enterprise Credit Decisioning Engine", version="1.0.0")

# Load our calibrated scorecard lookup matrix from Week 3
SCORECARD_PATH = "data/processed/scorecard_table.csv"
if os.path.exists(SCORECARD_PATH):
    scorecard_df = pd.read_csv(SCORECARD_PATH)
else:
    scorecard_df = None

# Define production applicant schema payload
class ApplicantData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: float
    PAY_0: float
    PAY_2: float
    PAY_3: float
    PAY_4: float
    PAY_5: float
    PAY_6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float

def lookup_points(variable_name: str, value: float) -> float:
    """Matches an applicant attribute value against its optimized scorecard bin to fetch points."""
    if scorecard_df is None:
        return 46.4  # Baseline fallback safety score
    
    # Filter matrix down to the single variable ruleset
    var_rules = scorecard_df[scorecard_df['Variable'] == variable_name]
    
    for _, row in var_rules.iterrows():
        bin_str = str(row['Bin'])
        
        # Handle structural system buckets
        if bin_str in ["Special", "Missing"]:
            continue
            
        # Parse standard mathematical interval string formats like [145000.00, 165000.00)
        cleaned = bin_str.replace("(", "").replace("[", "").replace(")", "").replace("]", "")
        parts = cleaned.split(",")
        
        lower = float(parts[0].strip())
        upper = float(parts[1].strip())
        
        if lower <= value < upper:
            return float(row['Points'])
            
    # Default to neutral system point if outside bounds
    return float(var_rules[var_rules['Bin'] == 'Missing']['Points'].fillna(46.4).iloc[0])

@app.post("/v1/underwrite")
def underwrite_applicant(applicant: ApplicantData):
    if scorecard_df is None:
        raise HTTPException(status_code=500, detail="Scorecard lookup sheet missing from production artifacts.")
        
    app_dict = applicant.model_dump()
    
    # STRATEGY RULE 1: HARD KNOCK-OUT RULES (Automated Cost-Saving Gate)
    # If an applicant is severely past due on recent payments (PAY_0 >= 3 means 90+ days late), reject immediately.
    if app_dict['PAY_0'] >= 3:
        return {
            "status": "REJECTED",
            "credit_score": None,
            "reason": "Hard Knock-Out Triggered: Applicant has critical recent delinquency status.",
            "pricing": None
        }
        
    # STRATEGY RULE 2: CALCULATE SYSTEM POINTS
    total_score = 0.0
    for variable, value in app_dict.items():
        total_score += lookup_points(variable, value)
        
    # Round to matching integer format
    final_credit_score = int(round(total_score))
    
    # STRATEGY RULE 3: RISK-BASED PRICING MATRIX
    if final_credit_score < 500:
        decision = "REJECTED"
        pricing = None
        reason = "Credit score falls below minimum portfolio risk tolerance threshold (< 500)."
    elif 500 <= final_credit_score < 650:
        decision = "APPROVED"
        reason = "Approved under Mid-Prime tier parameters."
        pricing = {
            "assigned_credit_limit": round(app_dict['LIMIT_BAL'] * 0.5, -2),  # Restrict exposure to 50%
            "offered_apr": "24.99%"                                          # High risk premium pricing
        }
    else:
        decision = "APPROVED"
        reason = "Approved under Super-Prime tier parameters."
        pricing = {
            "assigned_credit_limit": round(app_dict['LIMIT_BAL'] * 1.2, -2),  # Expand exposure to 120%
            "offered_apr": "11.25%"                                          # Competitive risk pricing
        }
        
    return {
        "status": decision,
        "credit_score": final_credit_score,
        "reason": reason,
        "pricing": pricing
    }