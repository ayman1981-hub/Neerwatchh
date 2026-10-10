# Neerwatchh
A fully local, no-paid-service early-warning prototype. Citizen reports in Tamil/English flow through three ML models — multilingual embedding + Logistic Regression symptom extraction, DBSCAN clustering, and an Isolation Forest anomaly score — under a rule engine (score > 0.70 AND ≥ 3 reports in 48h)
