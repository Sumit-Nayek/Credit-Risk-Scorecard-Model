# Credit-Risk-Scorecard-Model
(In progress)
# Enterprise Credit Risk Scorecard & Decision Engine

An end-to-end, production-grade credit scoring system modeled on consumer lending frameworks. The engine uses regularized statistical models and optimal Weight of Evidence (WoE) constraints to transform credit application metrics into a calibrated FICO-equivalent scorecard, executing real-time underwriting decisions through a containerized API layer.

## Technical Architecture Overview
* **Data Layer:** UCI Credit Card Portfolio Dataset (30,000 active customer records).
* **Feature Engineering:** Automated optimal binning constraints, monotonic risk filtering via Information Value (IV), and monotonic transformations.
* **Risk Model:** Regularized Logistic Regression estimator calibrated to an industry-standard Min-Max credit framework (300 to 850 score range).
* **Underwriting Layer:** FastAPI application serving real-time validation checks, hard knock-out constraints, dynamic exposure limits, and risk-adjusted APR pricing tiers.
* **MLOps Core:** Automated Population Stability Index (PSI) drift monitoring algorithms.

## Operational Financial Metrics (C-Suite Summary)
* **Optimal Cutoff Score Threshold:** 600
* **Projected Test Portfolio Net Revenue Capture:** $10,146,000.00
* **Portfolio Non-Performing Loan (NPL) Default Rate:** 0.75% 
* **Target Capital Efficiency Ratio:** 77.4% Approved (6,972 out of 9,000 applicants)

## Execution Guide
1. Install core requirements: `pip install -r requirements.txt`
2. Run data engineering tasks: `python src/data/data_ingestion.py`
3. Process optimal transformations: `python src/data/feature_engineering.py`
4. Run model calibration: `python src/models/scorecard_model.py`
5. Test optimization parameters: `python src/models/profit_optimization.py`
6. Initialize live decision API engine: `uvicorn src.api.app:app --port 8000`