Akudaikon Cyber-Loss Demo
Quantitative Monte Carlo Model for Cyber Risk and Control ROI

This Streamlit app demonstrates a Monte Carlo cyber-loss simulation that estimates Expected Annual Loss (EAL), Value at Risk (VaR), and visualizes the Loss Exceedance Curve (LEC).
It is designed to help organizations treat cybersecurity as a capital allocation problem—quantifying how each control reduces risk and justifying spend in CFO-native terms.

🎯 Goal

Model breach frequency and severity using probabilistic methods, simulate potential annual losses, and evaluate the Return on Security Investment (ROSI) for individual or bundled controls.
Each control has:

A defined annual cost

Likelihood and/or severity multipliers

A measurable impact on ΔEAL and ΔVaR

⚙️ How It Works
Component	Description	Notes
Frequency	Incidents per year modeled as Poisson(λ)	λ = mean annual event rate
Severity	(Beta fraction of customers) × (cost per record), capped by exposure	Replace with your trained severity model
Controls	Apply likelihood and/or severity multipliers (plus optional annual cost)	Simulate “what-if” security investments
Outputs	EAL, VaR95/99, LEC (log–log chart), ROI tables, CSV export	Compare baseline vs controlled scenarios
📊 Features

Monte Carlo engine — frequency ~ Poisson(λ); severity ~ heavy-tailed proxy

Controls — Server, Media, Error, External; user-defined multipliers and costs

Metrics — EAL, VaR95/99, and VaR-to-Net-Worth ratio

Charts — Loss Exceedance Curve (log–log tail risk view)

Exports — Download CSV of baseline & controlled simulations

ROI analysis — Per-control and marginal ROI when added to a bundle

🧩 Inputs (Sidebar)
Field	What It Means	Tips
NAICS sector	Seeds baseline breach probability and records per breach	Replace with sector priors or trained model hooks
Simulation trials	Number of Monte Carlo runs	10k for demos; more = smoother tails
Customers / records cap	Exposure limit for records affected	Source from NCUA 5300 or internal data
Net Worth ($)	Used to compute VaR / Net Worth ratios	Aligns with board & regulator reporting
Annual incident rate (λ)	Average incidents per year	Derived from IRP metrics or benchmarks
Cost per customer ($)	Average cost per impacted record	Replace with empirically derived estimates
Controls	Likelihood / severity multipliers	Use calibrated Δpropensity and Δseverity values
Control costs ($/yr)	Annualized cost per control	Enables ROI and portfolio optimization
Random seed	Ensures reproducibility	Fix for consistent charts and exports
