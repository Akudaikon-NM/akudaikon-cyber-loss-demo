akudaikon-cyber-loss-demo

Minimal Streamlit demo: Monte Carlo cyber-loss model with control toggles, EAL/VaR, and a Loss Exceedance Curve (LEC).
Defaults are synthetic placeholders; swap in your trained propensity and severity models when ready.

Goal: Treat security like capital allocation. Each control has a cost and reduces expected loss (ΔEAL) and tail risk (ΔVaR). Rank, bundle, and justify spend in CFO-native terms.

Features

Monte Carlo engine (frequency ~ Poisson(λ); heavy-tailed severity proxy)

Controls: Server, Media, Error, External (multipliers + annual costs)

Metrics: EAL, VaR95/99, VaR/Net-Worth ratio

Charts: Loss Exceedance Curve (log-log)

Exports: CSV of simulated losses (baseline & controlled)

ROI: Per-control isolated ROI and marginal ROI (adding to current bundle)

⚙️ How it works (short)

Frequency: Incidents ~ Poisson(λ)

Severity (demo): (Beta fraction of customers) × ($/customer), capped by customers

Controls: Apply likelihood/severity multipliers (and optional annual cost)

Outputs: EAL, VaR95/99, LEC, ROI tables, CSV export

Inputs (left panel)
Field	What it means	Tips
NAICS sector	Seeds starter priors (breach propensity, records per breach)	Replace with your sector priors / model hooks
Simulation trials	Number of Monte Carlo runs	10k is good for demos; more = smoother tails
Customers / records (cap)	Exposure at risk, used as cap for records exposed	Pull from 5300 Call Report or internal system of record
Net Worth ($)	Used for VaR / Net Worth ratios	Useful for board & regulator reporting
Annual incident rate λ	Mean incidents per year	Use IRP metrics or industry baselines
Cost per customer ($)	Severity scalar per impacted record	Replace with your empirically derived cost model
Controls	Apply multipliers for Server / Media / Error / External	Replace placeholder multipliers with calibrated Δpropensity / Δseverity
Control costs ($/yr)	Annualized cost per control	Enables ROI and marginal ROI analysis
Random seed	Reproducibility of results	Fix to repeat charts/exports
