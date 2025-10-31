Akudaikon Cyber-Loss Demo

Monte Carlo cyber-loss model in Streamlit with control toggles, EAL/VaR, and a Loss Exceedance Curve (LEC).
Defaults are synthetic placeholders—swap in your trained frequency/propensity and severity models when ready.

Goal

Treat security like capital allocation. Each control has a cost and reduces expected loss (ΔEAL) and tail risk (ΔVaR). Rank, bundle, and justify spend in CFO-native terms.

Features

Monte Carlo engine (frequency ~ Poisson(lambda); heavy-tailed severity proxy)

Controls (likelihood/severity multipliers + annual costs)

Metrics: EAL, VaR95/99, VaR-to-Net-Worth ratio

Charts: Loss Exceedance Curve (log–log)

Exports: CSV of simulated losses (baseline & controlled)

ROI: per-control (isolation) and marginal ROI (added to current bundle)

How it works (short)

Frequency: incidents per year ~ Poisson(lambda)

Severity (demo): (Beta fraction of customers) × (cost per record), capped by exposure

Controls: apply likelihood/severity multipliers (with optional annual cost)

Outputs: EAL, VaR95/99, LEC, ROI tables, CSV export

Inputs (sidebar)
Field	What it means	Tips
NAICS sector	Seeds baseline breach propensity and records per breach	Replace with sector priors or model hooks
Simulation trials	Number of Monte Carlo runs	10k is fine for demos; more = smoother tails
Customers / records cap	Exposure at risk; cap on records affected	Source from NCUA 5300 or internal system of record
Net Worth (USD)	Used for VaR / Net Worth ratios	Useful for board & regulator reporting
Annual incident rate (lambda)	Mean incidents per year	Use IRP metrics or industry baselines
Cost per record (USD)	Severity scalar per impacted record	Replace with empirically derived cost model
Controls	Likelihood / severity multipliers	Replace placeholder multipliers with calibrated deltas
Control costs (USD/yr)	Annualized cost per control	Enables ROI and marginal ROI analysis
Random seed	Reproducibility	Fix to repeat charts/exports
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
