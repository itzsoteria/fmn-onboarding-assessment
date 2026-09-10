# fmn-onboarding-assessment
Two AI-powered risk monitors for FMN: one forecasts SKU demand (backtested EWMA) to flag stockout/overstock risk with reorder logic; the other scores machine failure risk 24h ahead via RandomForest on sensor trends.

# Project 1 — Supply Chain Risk Monitor 
An AI-powered supply chain risk monitor. Forecasts SKU-level demand with a backtested EWMA model, flags SKUs as critical, at-risk, overstocked, or healthy using lead-time-aware reorder logic, and uses Claude to explain each flag in plain English and answer free-text questions grounded in live inventory data.

# Project 2 — Manufacturing Failure Risk Monitor
An AI-powered predictive maintenance tool. A RandomForest model, validated on a time-based holdout, scores each machine's failure risk for the next 24 hours from sensor trends and its own baseline, then uses Claude to explain why and answer free-text questions grounded in live sensor data.
